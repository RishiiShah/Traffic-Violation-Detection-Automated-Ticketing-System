# 🚦 Intelligent Traffic Surveillance System (ITSS)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![YOLOv12](https://img.shields.io/badge/YOLO-v12-orange?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Automating Traffic Compliance in India using Attention-Centric Real-Time Object Detection.**

---

## 📖 Table of Contents
- [Abstract](#-abstract)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Methodology & Logic](#-methodology--logic)
- [Tech Stack](#-tech-stack)
- [Performance Evaluation](#-performance-evaluation)
- [Installation](#-installation)
- [Usage](#-usage)
- [References & Citations](#-references--citations)

---

## 📄 Abstract

India’s rapid urbanization has led to a surge in vehicular density, rendering traditional manual traffic monitoring inefficient. Manual oversight is prone to human error, corruption, and inability to handle high-volume traffic 24/7. 

**ITSS (Intelligent Traffic Surveillance System)** is a vision-based automated solution designed to enforce traffic laws with high precision. By leveraging the **[YOLOv12 (You Only Look Once)](https://docs.ultralytics.com/models/yolo12/)** architecture, which introduces attention-centric mechanisms for superior real-time performance, this system detects complex violations such as red-light jumping, speeding, and helmet-less riding.

Furthermore, the system integrates an **OCR (Optical Character Recognition)** pipeline to extract license plates and communicates with a backend API to verify **Insurance** and **Pollution Control (PUC)** status, automatically generating e-challans for offenders.

---

## ✨ Key Features

### 1. Visual Violation Detection
* **🔴 Red Light Violation:** Detects the traffic signal state and identifies vehicles that cross the stop line while the signal is red.
* **⛑️ Helmet-less Rider Detection:** Classifies two-wheeler riders and specifically detects the absence of a helmet on both the rider and pillion.
* **⚡ Speeding Detection:** Uses perspective transformation and object tracking to estimate vehicle speed in real-time.

### 2. Automatic Number Plate Recognition (ANPR)
* **🔍 Plate Extraction:** High-accuracy cropping of license plates from moving vehicles.
* **📝 OCR Processing:** Converts plate images to text strings using advanced OCR engines.

### 3. Compliance & Enforcement
* **☁️ API Integration:** Queries a central database using the extracted plate number.
* **🚫 Document Verification:** Checks for valid Insurance and PUC certificates.
* **📩 E-Challan Generation:** Automatically creates a violation report with time, location, violation type, and evidence image.

---

## 🏗 System Architecture

The system operates on a pipeline approach:

1.  **Input Acquisition:** CCTV feed or pre-recorded video.
2.  **Object Detection (YOLOv12):** Detects classes: `Vehicle`, `License Plate`, `Traffic Light`, `Person`, `Helmet`.
3.  **Object Tracking:** Assigns unique IDs to vehicles across frames using **SORT/DeepSORT** to handle occlusion.
4.  **Violation Logic Module:**
    * *If Red Light AND Vehicle Center > Stop Line → Violation.*
    * *If Vehicle Speed > Speed Limit → Violation.*
    * *If Motorbike AND Head AND No Helmet → Violation.*
5.  **Post-Processing:**
    * Crop License Plate → Pass to OCR → Get String.
    * Send String to API → specific JSON response.
6.  **Output:** Overlay visuals on video and log data to CSV/Database.

---

## ⚙️ Methodology & Logic

### Why YOLOv12?
We utilize **YOLOv12** by Ultralytics. Unlike previous iterations, YOLOv12 introduces an **attention-centric architecture** that significantly improves feature extraction in complex urban environments (e.g., crowded Indian roads). It balances the speed of CNNs with the global context awareness of Transformers.
* *Reference:* [Ultralytics YOLOv12 Documentation](https://docs.ultralytics.com/models/yolo12/)

### Speed Estimation Logic
To calculate speed from a 2D video feed:
1.  **Perspective Transform:** We map the Region of Interest (ROI) on the road to a "bird's eye view."
2.  **Euclidean Distance:** Calculate the distance moved by the vehicle centroid between frames.
3.  **Formula:** $Speed = \frac{Distance (meters)}{Time (seconds)} \times 3.6 (km/h)$

---

## 🛠 Tech Stack

| Component | Tool/Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Core logic and scripting. |
| **Detection Model** | **YOLOv12** | Custom trained on Indian Traffic Dataset. |
| **Framework** | PyTorch / Ultralytics | Model training and inference. |
| **Computer Vision** | OpenCV | Image processing, perspective transforms. |
| **OCR** | EasyOCR / Tesseract | Text extraction from license plates. |
| **Tracking** | DeepSORT | Multi-object tracking for vehicle persistence. |
| **Backend** | Flask / FastAPI | Handling API requests for challan generation. |

---

## 📊 Performance Evaluation

The model was tested on a custom dataset comprising diverse Indian road scenarios (day, night, rain, high density).

* **Overall Accuracy:** **93.76%**
* **mAP@0.5:** 0.95
* **Inference Speed:** ~45 FPS on NVIDIA RTX 3060
* **OCR Accuracy:** 89% (Dependent on plate visibility)

---

## 📦 Installation

### Prerequisites
* Python 3.8 or higher
* CUDA-enabled GPU (Recommended for real-time performance)

### Steps

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/traffic-surveillance-system.git](https://github.com/yourusername/traffic-surveillance-system.git)
    cd traffic-surveillance-system
    ```

2.  **Create Virtual Environment (Optional but recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    # Install Ultralytics for YOLOv12
    pip install ultralytics

    # Install other requirements
    pip install -r requirements.txt
    ```

4.  **Download Weights**
    Place your trained `yolov12_custom.pt` file in the `weights/` directory.

---

## 💻 Usage

To run the system on a video file:

```bash
python main.py --source data/input_video.mp4 --weights weights/yolov12_custom.pt --conf 0.5
```

**Arguments:**

* `--source`: Path to video file or `0` for webcam.
* `--weights`: Path to the trained YOLOv12 model.
* `--conf`: Confidence threshold for detection.
* `--save-txt`: Save violation logs to a text file.

---

## 🔗 References & Citations

This project is built upon the cutting-edge research in object detection. If you use this repository or the YOLOv12 architecture in your research, please cite the original authors:

### YOLOv12 Architecture

```bibtex
@article{tian2025yolo12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}

```

### YOLOv12 Software

```bibtex
@software{yolo12,
  author = {Tian, Yunjie and Ye, Qixiang and Doermann, David},
  title = {YOLO12: Attention-Centric Real-Time Object Detectors},
  year = {2025},
  url = {[https://github.com/sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)},
  license = {AGPL-3.0}
}

```

### Acknowledgements

* **Ultralytics:** For the YOLO framework implementation. [Docs](https://docs.ultralytics.com/models/yolo12/)
* **OpenCV:** For image processing tools.

---

<div align="center">
<sub>Developed with ❤️ for safer roads in India.</sub>
</div>
