# Quantitative Impact of Image Denoising Prior to Edge Detection 🔍

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Shreyas-Reddy707/Computer-Vision/blob/main/Computer_Visions.ipynb)

## 📌 Project Overview
This project is an interactive computer vision case study that demonstrates and quantifies why image denoising is a mandatory pre-processing step before running gradient-based edge detection algorithms like the **Canny Edge Detector**. 

**Why we made this:** 
Edge detection algorithms mathematically rely on spatial intensity derivatives to find boundaries. Because differentiation acts as a high-pass filter, it heavily amplifies high-frequency variations in an image—specifically, sensor noise. If applied directly to raw data, the algorithm mistakes random noise spikes for actual structural boundaries, resulting in massive false-positive rates. Our team built this interactive workbench to visually and quantitatively prove how applying a low-pass filter (like Gaussian Blur) suppresses this noise and recovers true boundary accuracy.

---

## ✨ Key Features
* **Interactive UI:** Built with Gradio, allowing users to adjust noise levels and blur kernels in real-time.
* **Synthetic Noise Injection:** Simulates real-world sensor degradation using adjustable Additive Gaussian Noise.
* **Side-by-Side Comparison:** Compares direct edge detection (Control) against filtered edge detection (Experimental).
* **Quantitative Benchmarking:** Automatically computes standard evaluation metrics:
  * **False-Positive Edge Pixels:** Counts spurious noise artifacts.
  * **Edge Precision & F1-Score:** Measures the accuracy of the detected structural boundaries.
  * **SSIM (Structural Similarity Index):** Evaluates overall structural preservation.
* **High-Visibility Color Overlay:** Maps detected edges back to the clean ground-truth image for visual validation:
  * 🟢 **True Positives (Green):** Correctly identified boundaries.
  * 🔴 **False Positives (Red):** Phantom edges caused by noise.
  * 🔵 **False Negatives (Blue):** Missed boundaries (usually due to over-smoothing).

---

## 🛠️ Technology Stack
* **Language:** Python 3
* **Image Processing:** OpenCV (`cv2`)
* **Numerical & Statistical Computing:** NumPy, Pandas, Scikit-Image (`skimage.metrics`)
* **Web UI / Dashboard:** Gradio
* **Environment:** Google Colab

---

## 🚀 How to Run the Project

The easiest way to run this project is through Google Colab, as it handles all dependencies and environment setup automatically.

1. Click the **Open In Colab** badge at the top of this README.
2. Once the notebook opens, click on `Runtime` > `Run all` in the top menu (or press `Ctrl+F9`).
3. The script will automatically install the required libraries (`gradio`, `scikit-image`).
4. Scroll down to the output of the code cell. You will see a Gradio interface load directly in the notebook, along with a **public shareable URL** (e.g., `https://xxxx.gradio.live`).
5. **Upload an image**, and adjust the **Noise Intensity** and **Blur Kernel Size** sliders to observe the real-time impact on the edge metrics!

---
