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
git clone https://github.com/pathak-pragnya/model-compression-and-inference-optimization
cd model-compression-and-inference-optimization
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

## **Performance Metrics**
```
ONNX - Latency: 35.36 ms | Throughput: 28.28 inferences/sec
Keras - Latency: 102.69 ms | Throughput: 9.74 inferences/sec
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
