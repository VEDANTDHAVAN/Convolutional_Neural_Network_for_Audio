# 🎧 Audio CNN Visualizer

This project is a complete pipeline for training, deploying, and visualizing predictions of a **Convolutional Neural Network (CNN)** trained on environmental sound classification using the **ESC-50 dataset**. It includes a backend powered by **PyTorch + Modal** and a frontend built with **Next.js + TailwindCSS**.

---

## 📦 Features

- 📊 Train a deep CNN on ESC-50 dataset with mixup and spectrogram augmentation
- 🚀 Deploy the model with GPU support using [Modal](https://modal.com)
- 🎧 Upload audio files, view predictions and intermediate feature maps
- 🖼️ Visualize waveform, mel-spectrograms, and CNN layer activations in the browser

---

## 🗂️ Project Structure

```

.
├── audio-cnn-visual      # Frontend using T3-Stack
├── tensorboard_logs      # Log files of Modal Training Process
├── model.py              # CNN model with residual blocks
├── train.py              # Modal function to train model on ESC-50
├── main.py               # Modal-based inference server with audio + feature map visualization
├── requirements.txt      # Python dependencies
└── ...

````

---

## 🏁 Getting Started

### ⚙️ Backend Setup (Python 3.10+)

1. **Clone this repository**

```bash
git clone https://github.com/VEDANTDHAVAN/Convolutional_Neural_Network_for_Audio.git
cd audio-cnn-visualizer
````

2. **Install dependencies**

```bash
python -m venv .venv
.\.venv\Scripts\activate  # On Windows
source .venv/bin/activate # On macOS/Linux

pip install -r requirements.txt
```

3. **Train the model (optional)**

This uses GPU via Modal to train on ESC-50:

```bash
modal run train.py
```

The best model will be saved to a Modal volume (`esc-modal`).

4. **Run inference locally**

```bash
modal run main.py
```

This loads the model and sends a test audio file (`dogbark.wav`) as base64.

---

### 🖥️ Frontend Setup (Next.js)

1. **Install Node.js (v18+) and npm**
2. **Install dependencies**

```bash
npm install
```

3. **Run the development server**

```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

> ✅ The frontend connects to the Modal inference endpoint to visualize:
>
> * Top 3 predictions
> * Input waveform
> * Mel-spectrogram
> * Intermediate CNN feature maps

---

## 🧠 Model Details

* **Architecture**: Deep CNN with Residual Blocks
* **Input**: Log-Mel Spectrograms
* **Training Enhancements**:

  * `Mixup` augmentation
  * `Frequency` and `Time` masking
  * `Label smoothing`
  * `OneCycleLR` scheduler
* **Output**: Top-3 predicted classes + confidences

---

## 📊 Visualization Features

Returned from the inference API:

* `waveform`: Downsampled 1D audio array
* `input_spectogram`: 2D mel spectrogram
* `visualization`: Feature maps from all ResNet layers (`layer1.block0`, `layer2.block3`, etc.)

---

## 📁 Volumes (Modal)

* `/data`: ESC-50 dataset (downloaded in container)
* `/models`: Trained model + tensorboard logs

---

## 🚀 Deployment

This project uses [Modal](https://modal.com) for cloud-based training and inference on GPUs. Make sure to:

1. Install Modal CLI:
   `pip install modal`

2. Login via terminal:
   `modal token new`

3. Deploy/train via:
   `modal run train.py` or `modal run main.py`

---

## 🧪 Sample Test

Run the following to test locally:

```bash
modal run main.py
```

Expected Output:

```
Top Predictions!!
 - Dog Bark: 92.56%
 - Chirping Birds: 3.42%
 - Engine Idling: 1.12%
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* ESC-50 Dataset by Karol J. Piczak
* Modal for serverless GPU infrastructure
* PyTorch + torchaudio for audio processing

---

## 🤝 Contributing

Pull requests welcome! Let’s build smarter audio models together.

```
