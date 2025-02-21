# train_model.py

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from data_preparation import load_config, load_and_preprocess_data
from model import build_model
import os

def train_model(data_dir, img_size, batch_size, val_split, epochs, model_save_path, callbacks_config, model_config):
    """
    Trains the MobileNetV2-based model and saves the best-performing version.

    Args:
        data_dir (str): Path to the dataset directory.
        img_size (tuple): Target image size.
        batch_size (int): Batch size for training.
        val_split (float): Fraction of data reserved for validation.
        epochs (int): Number of training epochs.
        model_save_path (str): Path to save the best model.
        callbacks_config (dict): Configuration for callbacks.
        model_config (dict): Configuration for model building.

    Returns:
        history: Training history object.
    """
    print("[INFO] Starting data loading and preprocessing...")
    train_gen, val_gen = load_and_preprocess_data(data_dir, img_size, batch_size, val_split)

    print("[INFO] Building the model...")
    model = build_model(
        input_shape=img_size + (3,),
        num_classes=train_gen.num_classes,
        base_model_trainable=model_config.get('base_model_trainable', False),
        dropout_rate=model_config.get('dropout_rate', 0.3)
    )

    # Prepare callbacks
    print("[INFO] Setting up callbacks...")
    callbacks = []

    if callbacks_config.get('model_checkpoint', {}).get('enabled', True):
        checkpoint = ModelCheckpoint(
            model_save_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
        callbacks.append(checkpoint)
        print(f"[INFO] Model checkpoint callback enabled. Saving to: {model_save_path}")

    if callbacks_config.get('early_stopping', {}).get('enabled', True):
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=callbacks_config['early_stopping'].get('patience', 5),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        print("[INFO] Early stopping callback enabled.")

    if callbacks_config.get('reduce_lr', {}).get('enabled', True):
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=callbacks_config['reduce_lr'].get('factor', 0.2),
            patience=callbacks_config['reduce_lr'].get('patience', 3),
            min_lr=callbacks_config['reduce_lr'].get('min_lr', 1e-6),
            verbose=1
        )
        callbacks.append(reduce_lr)
        print("[INFO] Reduce learning rate callback enabled.")

    # Train the model
    print(f"[INFO] Starting training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    print("[INFO] Training completed.")
    return history

if __name__ == "__main__":
    print("[INFO] Starting training script...")

    try:
        config = load_config("config.yaml")

        data_config = config['data']
        train_config = config['train']
        callbacks_config = config['callbacks']
        model_config = config['model']

        history = train_model(
            data_dir=data_config['data_dir'],
            img_size=tuple(data_config['img_size']),
            batch_size=data_config['batch_size'],
            val_split=data_config['validation_split'],
            epochs=train_config['epochs'],
            model_save_path=train_config['model_save_path'],
            callbacks_config=callbacks_config,
            model_config=model_config
        )

    except Exception as e:
        print(f"[ERROR] Training script failed: {e}")
