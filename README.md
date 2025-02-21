# **README.md**

## **Model Compression and Evaluation Pipeline**
This repository provides a comprehensive pipeline to train, compress, convert, and evaluate deep learning models using **MobileNetV2** as the base architecture. The pipeline includes:
- Data preparation with augmentation.
- Model definition with configurable parameters.
- Model training with callbacks.
- Model compression (pruning and quantization).
- Model conversion to ONNX format.
- Evaluation of latency and throughput for both Keras and ONNX models.

---

## **Project Structure**
```
├── data_preparation.py       # Data loading and augmentation
├── model_definition.py       # MobileNetV2 model creation
├── train_model.py            # Training with configurable callbacks
├── model_compression.py      # Pruning and quantization of the trained model
├── evaluate_model.py         # Inference benchmarking and ONNX conversion
├── config.yaml               # Configuration for all stages
└── README.md                 # Project overview and usage instructions
```

---

## **Installation**
1. **Clone the repository:**
```bash
git clone <repository-url>
cd <repository-directory>
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## **Configuration**
All parameters are managed via the `config.yaml` file.

### Sample `config.yaml`:
```yaml
data:
  data_dir: "path_to_dataset"
  img_size: [224, 224]
  batch_size: 32
  validation_split: 0.2
  visualize_samples: true
  num_samples: 5

model:
  input_shape: [224, 224, 3]
  num_classes: 10
  base_model_trainable: false
  dropout_rate: 0.3

train:
  epochs: 20
  model_save_path: "best_model.keras"

callbacks:
  model_checkpoint:
    enabled: true
  early_stopping:
    enabled: true
    patience: 5
  reduce_lr:
    enabled: true
    factor: 0.2
    patience: 3
    min_lr: 1e-6

compression:
  model_path: "best_model.keras"
  save_path: "compressed_model.keras"
  apply_pruning: true
  apply_quantization: true
  pruning_params:
    pruning_schedule:
      initial_sparsity: 0.1
      final_sparsity: 0.5
      begin_step: 0
      end_step: 1000

evaluation:
  keras_model_path: "compressed_model.keras"
  onnx_model_path: "compressed_model.onnx"
  convert_to_onnx: true
  input_shape: [224, 224, 3]
  num_iterations: 100
```

---

## **Usage**
### 1. **Data Preparation**
```bash
python data_preparation.py --config config.yaml
```
- Loads data with augmentation.
- Optionally visualizes sample images.

---

### 2. **Model Definition**
```bash
python model_definition.py --config config.yaml
```
- Builds a MobileNetV2-based model.
- Freezes or unfreezes layers based on the config.

---

### 3. **Training**
```bash
python train_model.py --config config.yaml
```
- Trains the model with early stopping and learning rate reduction.
- Saves the best-performing model.

---

### 4. **Model Compression**
```bash
python model_compression.py --config config.yaml
```
- Applies pruning and quantization.
- Saves the compressed model.

---

### 5. **Evaluation**
```bash
python evaluate_model.py --config config.yaml
```
- Benchmarks latency and throughput of Keras and ONNX models.
- Validates ONNX model structure.

---

## **Sample Output**
```
[RESULT] Keras - Latency: 12.34 ms | Throughput: 81.20 inferences/sec
[RESULT] ONNX  - Latency: 8.56 ms | Throughput: 116.89 inferences/sec
```

---

## **Requirements**
- Python 3.7+
- TensorFlow >= 2.8
- tf2onnx
- onnx
- onnxruntime
- tensorflow-model-optimization
- numpy
- matplotlib
- pyyaml

---

## **Contributing**
Contributions are welcome! Feel free to submit issues or pull requests.
