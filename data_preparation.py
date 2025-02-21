# data_preparation.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_and_preprocess_data(data_dir, img_size=(224, 224), batch_size=32, val_split=0.2):
    """
    Loads and preprocesses image data from the specified directory.

    Args:
        data_dir (str): Path to the dataset directory.
        img_size (tuple): Target size for image resizing.
        batch_size (int): Batch size for data generators.
        val_split (float): Validation split ratio.

    Returns:
        train_generator, val_generator: Data generators for training and validation.
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=val_split,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        fill_mode="nearest",
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_generator, val_generator


def visualize_samples(generator, num_samples=5):
    """
    Displays sample images from the data generator.

    Args:
        generator (DirectoryIterator): Data generator.
        num_samples (int): Number of samples to display.
    """
    import matplotlib.pyplot as plt

    images, labels = next(generator)
    plt.figure(figsize=(15, 5))

    for i in range(num_samples):
        ax = plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        plt.title(f"Class: {generator.class_indices}\nLabel: {labels[i].argmax()}")
        plt.axis("off")

    plt.show()


if __name__ == "__main__":
    # Load configuration
    config = load_config("config.yaml")

    DATA_DIR = config['data']['data_dir']
    IMG_SIZE = tuple(config['data']['img_size'])
    BATCH_SIZE = config['data']['batch_size']
    VAL_SPLIT = config['data']['val_split']

    # Load and preprocess data
    train_gen, val_gen = load_and_preprocess_data(DATA_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT)

    print(f"Train samples: {train_gen.samples}, Validation samples: {val_gen.samples}")

    # Visualize samples if enabled
    if config['data'].get('visualize_samples', False):
        visualize_samples(train_gen, num_samples=config['data'].get('num_samples', 5))
