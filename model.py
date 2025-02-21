import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import yaml
import os

def load_config(config_path="config.yaml"):
    """
    Loads configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration parameters.
    """
    print(f"[INFO] Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"[INFO] Configuration loaded successfully: {config}")
        return config
    except FileNotFoundError:
        print(f"[ERROR] Configuration file not found at {config_path}.")
        raise
    except yaml.YAMLError as e:
        print(f"[ERROR] Failed to parse YAML file: {e}")
        raise

def build_model(input_shape=(224, 224, 3), num_classes=10, base_model_trainable=False, dropout_rate=0.3):
    """
    Builds and compiles a MobileNetV2-based model with custom classification layers.

    Args:
        input_shape (tuple): Input shape of the images.
        num_classes (int): Number of classes for classification.
        base_model_trainable (bool): Whether to make the base MobileNetV2 layers trainable.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        model (tf.keras.Model): Compiled Keras model.
    """
    print(f"[INFO] Building MobileNetV2 model with input shape {input_shape} and {num_classes} classes.")

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = base_model_trainable

    print(f"[INFO] Base model layers {'trainable' if base_model_trainable else 'frozen'}.")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate, name="dropout_layer")(x)
    outputs = Dense(num_classes, activation="softmax", name="output_layer")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("[INFO] Model compiled successfully.")
    return model

if __name__ == "__main__":
    print("[INFO] Starting model definition script...")
    try:
        config = load_config("config.yaml")

        input_shape = tuple(config['model']['input_shape'])
        num_classes = config['model']['num_classes']
        base_model_trainable = config['model'].get('base_model_trainable', False)
        dropout_rate = config['model'].get('dropout_rate', 0.3)

        model = build_model(
            input_shape=input_shape,
            num_classes=num_classes,
            base_model_trainable=base_model_trainable,
            dropout_rate=dropout_rate
        )
        model.summary()

    except Exception as e:
        print(f"[ERROR] Model definition script failed: {e}")
