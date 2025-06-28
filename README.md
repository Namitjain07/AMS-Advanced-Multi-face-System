# 🚀 AMS - Advanced Multi-face System

A powerful and modern facial detection and recognition system that enhances RetinaFace's limitations using an innovative **9-segment image technique**, multiple face matching libraries, and advanced upscaling methods.

---
## 🎥 Project Demo

Watch the demo below to see AMS in action:

[![AMS Demo](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID)

## 📸 Overview

**AMS (Advanced Multi-face System)** is designed to tackle the challenge of detecting and recognizing multiple human faces in images — especially when some are far from the camera or blurred.

- 🔍 **Base Detection:** Utilizes [RetinaFace](https://github.com/serengil/retinaface) for initial face detection.
- 🧠 **Enhanced Accuracy:** Introduces a unique **9-segment image technique** that improves detection rates significantly.
- 🧬 **Face Matching:** Employs multiple face matching libraries like `ArcFace`, `Dlib`, `Mediapipe`, and `CV2` for robust identity verification.
- 🧼 **Duplicate Removal:** Detects and removes duplicate faces across segments with feature comparison.
- 🔍 **Face Upscaling:** Applies various face unblurring and super-resolution algorithms.

---

## 🧩 9-Segment Image Technique

To address RetinaFace’s limitations on small or distant faces, AMS splits the image into **9 overlapping blocks** and applies detection on each:

### 🎯 Original Matrix
```

|  1 |  2 |  3 |  4 |
|  5 |  6 |  7 |  8 |
|  9 | 10 | 11 | 12 |
| 13 | 14 | 15 | 16 |

```

---

### 🧠 Segment Blocks

#### 🔹 Block 1
```

|  1 |  2 |
|  5 |  6 |

```

#### 🔹 Block 2
```

|  3 |  4 |
|  7 |  8 |

```

#### 🔹 Block 3
```

|  9  | 10 |
| 13 | 14 |

```

#### 🔹 Block 4
```

| 11 | 12 |
| 15 | 16 |

```

#### 🔹 Block 5
```

|  2 |  3 |
|  6 |  7 |

```

#### 🔹 Block 6
```

|  5 |  6 |
|  9 | 10 |

```

#### 🔹 Block 7
```

|  7 |  8 |
| 11 | 12 |

```

#### 🔹 Block 8
```

| 10 | 11 |
| 14 | 15 |

```

#### 🔹 Block 9
```

|  6 |  7 |
| 10 | 11 |

````

> These segmented crops are **upscaled** back to original size for detection, significantly improving overall accuracy.

---

## 🧬 Libraries Used

- 🎯 **Face Detection:** `RetinaFace`
- 🧑‍🤝‍🧑 **Face Matching:** `ArcFace`, `Dlib`, `Mediapipe`, `CV2`
- 🔁 **Feature Verification:** Facial mesh & feature vector comparison
- 📈 **Upscaling:** Multiple super-resolution techniques (WIP on glasses!)

---

## ⚙️ Setup Instructions

### 🐍 1. Clone the repository
```bash
git clone https://github.com/Namitjain07/AMS-Advanced-Multi-face-System.git
cd AMS-Advanced-Multi-face-System
````

### 💻 2. Create and activate a new conda environment

```bash
conda create -n AMS python=3.10
conda activate AMS
```

### 📦 3. Install dependencies

```bash
pip install \
retina-face==0.0.17 numpy==1.26.4 gdown==5.2.0 Pillow==11.1.0 \
opencv-python==4.10.0.84 tensorflow==2.17.1 beautifulsoup4==4.12.3 \
filelock==3.17.0 requests[socks]==2.32.3 tqdm==4.67.1 absl-py==1.4.0 \
astunparse==1.6.3 flatbuffers==25.1.21 gast==0.6.0 google-pasta==0.2.0 \
h5py==3.12.1 libclang==18.1.1 ml-dtypes==0.4.1 opt-einsum==3.4.0 \
packaging==24.2 protobuf==4.25.6 setuptools==75.1.0 six==1.17.0 \
termcolor==2.5.0 typing-extensions==4.12.2 wrapt==1.17.2 grpcio==1.70.0 \
tensorboard==2.17.1 keras==3.5.0 tensorflow-io-gcs-filesystem==0.37.1 \
wheel==0.45.1 rich==13.9.4 namex==0.0.8 optree==0.14.0 charset-normalizer==3.4.1 \
idna==3.10 urllib3==2.3.0 certifi==2024.12.14 markdown==3.7 \
tensorboard-data-server==0.7.2 werkzeug==3.1.3 soupsieve==2.6 PySocks==1.7.1 \
MarkupSafe==3.0.2 markdown-it-py==3.0.0 pygments==2.18.0 mdurl==0.1.2 \
mediapipe==0.10.9 pyheif
```

---

## 🛠️ Roadmap

* [x] Implement 9-segment strategy
* [x] Add face matching & duplicate removal
* [x] Test and compare upscaling methods
* [ ] Improve handling of spectacles
* [ ] Integrate with real-time video input
* [ ] Build minimal UI for evaluation

---

## 👨‍💻 Contributions

Want to help? Feel free to open issues or pull requests!

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## 💬 Feedback

We're actively improving AMS. If you encounter issues or have suggestions for better face detection (especially for glasses!), open an issue or reach out.

---

🌟 **Star this project** if you find it helpful!
