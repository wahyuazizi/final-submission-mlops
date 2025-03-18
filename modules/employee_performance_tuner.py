from typing import Any, Dict, NamedTuple, Text
import tensorflow as tf
import keras_tuner as kt
import tensorflow_transform as tft
from keras_tuner.engine import base_tuner
from tfx.components.trainer.fn_args_utils import FnArgs

from modules.employee_performance_transform import (
    LABEL_KEY,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    transformed_name
)

epochs = 10

TunerFnResult = NamedTuple("TunerFnResult", [(
    "tuner", base_tuner.BaseTuner), ("fit_kwargs", Dict[Text, Any])])


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_fn(
        file_pattern,
        tf_transform_output,
        num_epochs,
        batch_size=64
) -> tf.data.Dataset:

    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    # Create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    ).repeat(64)

    return dataset


def model_builder(hp):
    # num_layer = hp.Int("num_layer", min_value=1, max_value=5, step=1)
    # dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    fc_units = hp.Int("fc_units", min_value=16, max_value=64, step=16)
    lr = hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])

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
    x = tf.keras.layers.Dense(fc_units, activation="relu")(x)
    x = tf.keras.layers.Dense(fc_units, activation="relu")(x)

    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    

    model = tf.keras.Model(inputs=input_features, outputs=outputs)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    model.summary()

    return model


def tuner_fn(fn_args):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_sparse_categorical_accuracy",
        mode="max",
        verbose=1,
        patience=5
    )

    tuner = kt.Hyperband(
        lambda hp: model_builder(hp),
        objective="val_sparse_categorical_accuracy",
        max_epochs=25,
        factor=3,
        directory=fn_args.working_dir,
        project_name="kt_hyperband"
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [early_stop],
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps
        }
    )
