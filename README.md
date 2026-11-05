# Real-Time Multimodal Binary Decision Model

Zero-shot contrastive visual reasoning and decision-making over live webcam streams using dynamic multimodal routing (CLIP, X-CLIP, and FaceMesh/Audio).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cilyde/BinaryModel/blob/main/BinaryModel.ipynb)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/)

---

## 📌 Overview

**Binary Decision Model** enables low-latency visual classification and question answering directly on live video streams. Rather than relying on static pre-trained heads, it operates on **explicit contrastive zero-shot pairs** (`Positive vs Negative`) to evaluate visual conditions on the fly without retraining.

### 🔀 Dynamic Multimodal Routing
The router analyzes the semantics of each natural language query and dispatches video frames to specialized inference engines:

- **Static Visual Traits** (`clean shaven vs bearded face`, `wearing glasses vs no glasses`):  
  Evaluated using **CLIP (ViT-L/14)** with Exponential Moving Average (EMA) smoothing to eliminate single-frame flicker.
- **Dynamic Action & Motion** (`waving vs still`, `jumping vs standing`):  
  Evaluated using **X-CLIP** over an 8-frame sliding spatio-temporal window.
- **Speech & Facial Motion** (`talking vs silent`):  
  Evaluated using **MediaPipe FaceMesh** mouth landmark variance fused with microphone WebAudio RMS energy.
- **Manual Overrides**:  
  Force specific routing using prefixes: `static: ...`, `video: ...`, or `talk: ...`.

---

## 🚀 Quickstart

1. Click the **Open in Colab** badge above or open [`BinaryModel.ipynb`](BinaryModel.ipynb).
2. Set runtime type to GPU (**Runtime** ➔ **Change runtime type** ➔ **T4 GPU**).
3. Run the setup and model cells, then launch the interactive console to start streaming from your webcam.
