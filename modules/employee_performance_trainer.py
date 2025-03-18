import os
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import tensorflow_transform as tft

from modules.employee_performance_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)


def get_model(hp, show_summary=True):
    """
    This function defines a Keras model and returns
    Keras object
    """

    # One-hot categorical features
    input_features = []
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=transformed_name(key))
        )

    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    concat = tf.keras.layers.concatenate(input_features)
    x = tf.keras.layers.Dense(256, activation='relu')(concat)
    x = tf.keras.layers.Dense(hp.get("fc_units"), activation="relu")(x)
    x = tf.keras.layers.Dense(hp.get("fc_units"), activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.get("lr")),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    if show_summary:
        model.summary()

    return model


def gzip_reader_fn(filenames):
    """Loads comporessed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}
    return serve_tf_examples_fn


def input_fn(file_pattern, tf_transform_output, batch_size=64):
    """Generates features and labels for tuning/training.
    Args:
        file_pattern: input tfrecord file pattern.
        tf_transform_output: A TFTransformOutput.
        batch_size: representing the number of consecutive elements of
        returned dataset to combine in a single batch
    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices.
    """

    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY)
    ).repeat(64)

    return dataset


def run_fn(fn_args):
    """Train the model based on given args.
    Args:
    fn_args: Holds args used to train the model as name/value pairs.
    """

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_datset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_datset = input_fn(fn_args.eval_files, tf_transform_output, 64)

    hp = fn_args.hyperparameters.get("values")
    model = get_model(hp, show_summary=True)

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_sparse_categorical_accuracy",
        mode="max",
        verbose=1,
        patience=10
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor="val_sparse_categorical_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True
    )

    model.fit(
        train_datset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_datset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, early_stop, model_checkpoint],
        epochs=15
    )

    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name="examples"
            )
        ),
    }

    model.save(
        fn_args.serving_model_dir, save_format="tf", signatures=signatures
    )

    plot_model(
        model,
        to_file='images/model_plot.png',
        show_shapes=True,
        show_layer_names=True
    )
