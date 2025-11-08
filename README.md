# 🧠 Optimized BLIP-Based Offline Lens

A powerful **AI-driven image captioning tool** that uses your webcam and the **BLIP (Bootstrapped Language-Image Pretraining)** model to generate accurate, human-like descriptions of what the camera sees — all **offline** and optimized for GPU acceleration.

---

## 🚀 Overview

This project captures frames directly from your **webcam**, feeds them into the **Salesforce BLIP** model, and returns **descriptive captions** of the image.

It automatically detects and uses **GPU (CUDA)** if available, falls back to **CPU** otherwise, and uses **beam search** for more accurate natural-language descriptions.

---

## ✨ Features

- ⚡ **GPU auto-detection** (uses CUDA if available)  
- 🔍 **Beam search decoding** for improved caption accuracy  
- 📷 **Live webcam capture** — press a key to describe a scene  
- 🧩 **Optimized 384×384 preprocessing** (BLIP’s native input size)  
- 🪶 **Offline mode** — once the model is downloaded, no internet required  
- 🧠 Uses Hugging Face’s **Salesforce/blip-image-captioning-base** model  

---

## 🧰 Tech Stack

- **Python 3.8+**
- **PyTorch** — deep learning backend  
- **Transformers** — BLIP model pipeline  
- **OpenCV** — real-time webcam capture  
- **Pillow (PIL)** — image handling and conversion  

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/sanjai-cpu/BLIP-Offline-Lens.git
cd BLIP-Offline-Lens
2️⃣ Install dependencies
bash
Copy code
pip install torch torchvision torchaudio
pip install transformers
pip install opencv-python Pillow accelerate
(You may need to install torch with CUDA if you have a GPU — check PyTorch.org)

▶️ Usage
Run the script
bash
Copy code
python blip_offline_lens.py
Controls
Key	Action
s	Capture an image and generate a description
q	Quit the application

Example Output
arduino
Copy code
Initializing webcam...
⏳ Loading BLIP model on GPU...
✅ Model loaded successfully.

Press 's' to capture an image and get a description.
Press 'q' to quit the application.

📸 Capturing image...
📝 Description: a man wearing headphones sitting in front of a computer
🧠 How It Works
Webcam Input → Captures a live frame via OpenCV.

Preprocessing → Converts frame to RGB and resizes to 384×384 (BLIP’s expected input).

Model Inference → BLIP generates a text description using beam search.

Output → The generated caption is printed to the console.

