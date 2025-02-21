import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import time
import onnx
import onnxruntime as ort
import tf2onnx
from data_preparation import load_config
import os

def convert_to_onnx(model_path, onnx_save_path):
    """
    Converts a Keras model to ONNX format.

    Args:
        model_path (str): Path to the Keras model.
        onnx_save_path (str): Path to save the converted ONNX model.

    Returns:
        str: Path to the saved ONNX model.
    """
    print(f"[INFO] Loading Keras model from: {model_path}")
    model = load_model(model_path)

    print("[INFO] Converting Keras model to ONNX format...")
    input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input")]

    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)
    onnx.save(onnx_model, onnx_save_path)

    print(f"[INFO] ONNX model saved to: {onnx_save_path}")
    return onnx_save_path

def validate_onnx(onnx_model_path):
    """
    Validates the ONNX model.

    Args:
        onnx_model_path (str): Path to the ONNX model.
    """
    print(f"[INFO] Validating ONNX model at: {onnx_model_path}")
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print("[INFO] ONNX model validation successful.")

def evaluate_keras_model(model_path, input_shape=(224, 224, 3), num_iterations=100):
    """
    Evaluates the Keras model's inference latency and throughput.

    Args:
        model_path (str): Path to the Keras model.
        input_shape (tuple): Input shape for inference.
        num_iterations (int): Number of iterations for benchmarking.

    Returns:
        latency (float): Average latency in milliseconds.
        throughput (float): Inferences per second.
    """
    print(f"[INFO] Evaluating Keras model from: {model_path}")
    model = load_model(model_path)
    input_data = np.random.rand(1, *input_shape).astype(np.float32)

    print(f"[INFO] Running {num_iterations} inferences for benchmarking...")
    start = time.time()
    for _ in range(num_iterations):
        model.predict(input_data, verbose=0)
    total_time = time.time() - start

    latency = (total_time / num_iterations) * 1000  # ms
    throughput = num_iterations / total_time

    print(f"[RESULT] Keras - Latency: {latency:.2f} ms | Throughput: {throughput:.2f} inferences/sec")
    return latency, throughput

def evaluate_onnx_model(onnx_model_path, input_shape=(224, 224, 3), num_iterations=100):
    """
    Evaluates the ONNX model's inference latency and throughput.

    Args:
        onnx_model_path (str): Path to the ONNX model.
        input_shape (tuple): Input shape for inference.
        num_iterations (int): Number of iterations for benchmarking.

    Returns:
        latency (float): Average latency in milliseconds.
        throughput (float): Inferences per second.
    """
    print(f"[INFO] Evaluating ONNX model from: {onnx_model_path}")
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    input_data = np.random.rand(1, *input_shape).astype(np.float32)

    print(f"[INFO] Running {num_iterations} inferences for benchmarking...")
    start = time.time()
    for _ in range(num_iterations):
        session.run(None, {input_name: input_data})
    total_time = time.time() - start

    latency = (total_time / num_iterations) * 1000  # ms
    throughput = num_iterations / total_time

    print(f"[RESULT] ONNX - Latency: {latency:.2f} ms | Throughput: {throughput:.2f} inferences/sec")
    return latency, throughput

if __name__ == "__main__":
    print("[INFO] Starting model evaluation script...")

    try:
        config = load_config("config.yaml")
        evaluation_config = config['evaluation']

        keras_model_path = evaluation_config['keras_model_path']
        onnx_model_path = evaluation_config['onnx_model_path']
        input_shape = tuple(evaluation_config.get('input_shape', [224, 224, 3]))
        num_iterations = evaluation_config.get('num_iterations', 100)

        # Evaluate Keras model
        keras_latency, keras_throughput = evaluate_keras_model(
            keras_model_path, input_shape=input_shape, num_iterations=num_iterations
        )

        # Convert to ONNX and validate
        if evaluation_config.get('convert_to_onnx', True):
            onnx_model_path = convert_to_onnx(keras_model_path, onnx_model_path)
            validate_onnx(onnx_model_path)

        # Evaluate ONNX model
        onnx_latency, onnx_throughput = evaluate_onnx_model(
            onnx_model_path, input_shape=input_shape, num_iterations=num_iterations
        )

        print("\n[FINAL RESULTS]")
        print(f"Keras - Latency: {keras_latency:.2f} ms | Throughput: {keras_throughput:.2f} inferences/sec")
        print(f"ONNX  - Latency: {onnx_latency:.2f} ms | Throughput: {onnx_throughput:.2f} inferences/sec")

    except Exception as e:
        print(f"[ERROR] Model evaluation script failed: {e}")
