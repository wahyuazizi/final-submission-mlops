import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    "BusinessUnit": 10,
    "EmployeeStatus": 2,
    "EmployeeType": 3,
    "PayZone": 3,
    "EmployeeClassificationType": 3,
    "DepartmentType": 6,
    "GenderCode": 2,
    "RaceDesc": 5,
    "MaritalDesc": 4,
    "TrainingType": 2,
    "TrainingOutcome": 4
}

NUMERICAL_FEATURES = [
    "CurrentEmployeeRating",
    "EngagementScore",
    "SatisfactionScore",
    "Work-LifeBalanceScore",
    "TrainingDurationDays",
    "TrainingCost",
    "Age"
]

LABEL_KEY = "PerformanceScore"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def convert_num_to_one_hot(label_tensor, num_labels=4):
    """
    Convert a label into a one-hot vector.
    
    Args:
        label_tensor: Tensor berisi label dalam bentuk angka (0, 1, 2, 3)
        num_labels: Jumlah unik kelas (default: 4)
    
    Returns:
        One-hot encoded tensor dengan shape [-1, num_labels]
    """

    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    Args:
        inputs: map from feature keys to raw features
    Returns:
        outputs: map from feature keys to transformed features
    """

    outputs = {}

    for key in CATEGORICAL_FEATURES:
        dim = CATEGORICAL_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=dim + 1
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)
    return outputs