import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
from data_preparation import load_config
import os

def apply_pruning(model, pruning_params=None):
    """
    Applies pruning to the provided Keras model.

    Args:
        model (tf.keras.Model): Original model to be pruned.
        pruning_params (dict): Pruning schedule parameters.

    Returns:
        pruned_model (tf.keras.Model): Pruned Keras model.
    """
    print("[INFO] Applying pruning to the model...")

    if pruning_params is None:
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.5,
                begin_step=0,
                end_step=1000
            )
        }

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruned_model = prune_low_magnitude(model, **pruning_params)

    print("[INFO] Pruning applied successfully.")
    return pruned_model

def apply_quantization(model):
    """
    Applies post-training quantization to the provided model.

    Args:
        model (tf.keras.Model): Model to quantize.

    Returns:
        quantized_model (tf.keras.Model): Quantized model.
    """
    print("[INFO] Applying post-training quantization...")
    quantize_model = tfmot.quantization.keras.quantize_model
    quantized_model = quantize_model(model)
    print("[INFO] Quantization applied successfully.")
    return quantized_model

def save_compressed_model(model, save_path):
    """
    Saves the compressed model to the specified path.

    Args:
        model (tf.keras.Model): Model to save.
        save_path (str): File path to save the model.
    """
    print(f"[INFO] Saving compressed model to: {save_path}")
    model.save(save_path)
    print("[INFO] Compressed model saved successfully.")

def compress_model(model_path, save_path, compression_config):
    """
    Compresses a model using pruning and quantization.

    Args:
        model_path (str): Path to the model to compress.
        save_path (str): Path to save the compressed model.
        compression_config (dict): Configuration for compression steps.

    Returns:
        None
    """
    print(f"[INFO] Loading model from: {model_path}")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model file '{model_path}' not found.")
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path)
    print("[INFO] Model loaded successfully.")

    if compression_config.get('apply_pruning', False):
        pruning_params = compression_config.get('pruning_params', None)
        model = apply_pruning(model, pruning_params)

    if compression_config.get('apply_quantization', False):
        model = apply_quantization(model)

    save_compressed_model(model, save_path)

if __name__ == "__main__":
    print("[INFO] Starting model compression script...")
    
    try:
        config = load_config("config.yaml")
        compression_config = config['compression']

        compress_model(
            model_path=compression_config['model_path'],
            save_path=compression_config['save_path'],
            compression_config=compression_config
        )

    except Exception as e:
        print(f"[ERROR] Model compression script failed: {e}")
