# ⚾ MLB Pitch Type Classification from Video using CNN-LSTM

This project focuses on classifying baseball pitch types from short video clips using a deep learning architecture that combines **Convolutional Neural Networks (CNNs)** for spatial feature extraction and **Long Short-Term Memory (LSTM)** networks for temporal sequence modeling.

The goal is to learn motion and visual patterns across video frames to predict the type of pitch thrown (e.g., Fastball, Curveball, Changeup, etc.).

---

## 🚀 Project Overview

- **Task:** Multi-class video classification  
- **Input:** Short segmented pitch videos  
- **Output:** Predicted pitch type label  
- **Model:** ResNet18 (CNN) + LSTM (temporal modeling)  
- **Framework:** PyTorch  

Each video is treated as a sequence of frames:
- CNN extracts spatial features per frame
- LSTM models motion and temporal dynamics across frames
- Final classifier predicts pitch category

---

## 📁 Dataset

This project uses segmented video clips and metadata derived from the **MLB YouTube Dataset**:

🔗 Dataset Repo: https://github.com/piergiaj/mlb-youtube

### Expected Directory Structure

```
data/
 └── segmented_videos/
      ├── video_001.mp4
      ├── video_002.mp4
      ├── ...
      └── metadata.csv
```

### Metadata File

`metadata.csv` contains :

| Column Name | Description |
|--------|------------|
| `video_id` | Video file name |
| `clip_id` | Clip ID from each video |
| `start_time` | Start time of the clip with respect to the video |
| `end_time` | End time of the clip with respect to the video |
| `duration` | Length of the clip |
| `pitch_type` | Label for pitch class |
| `subset` | Whether data used for training or testing |

Among other columns

---

## 🧠 Model Architecture

### CNN Backbone
- Pretrained **ResNet18**
- Final classification layer removed
- Outputs 512-D feature vector per frame

### Temporal Modeling
- **LSTM** processes frame-level embeddings
- Captures motion patterns across frames

### Final Classifier
- Fully connected layer on top of LSTM output
- Softmax via CrossEntropyLoss

```
Video → Frames → CNN → Feature Sequence → LSTM → FC → Pitch Class
```

---

## 🏋️ Training Pipeline

- Frame extraction using OpenCV
- Uniform sampling of frames per clip
- Standard image normalization
- Mini-batch training using PyTorch DataLoader

### Loss & Optimization
- **Loss:** Cross Entropy Loss
- **Optimizer:** Adam
- **Metrics:** Accuracy

Training history includes:
- Training loss
- Validation loss
- Training accuracy
- Validation accuracy

Loss and accuracy curves are plotted after training.

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install torch torchvision opencv-python pandas numpy tqdm matplotlib seaborn
```

(If using Google Colab, most packages are preinstalled.)

---

### 2. Mount Google Drive (Colab Only)

The notebook assumes dataset access from Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Update dataset paths if running locally.

---

### 3. Configure Paths

Update paths in the notebook:

```python
CONFIG = {
    "data_dir": "/content/drive/MyDrive/segmented_videos/",
    "metadata_path": "/content/drive/MyDrive/segmented_videos/metadata.csv"
}
```

---

### 4. Train the Model

Run all notebook cells:

```
Runtime → Run All
```

Training progress is displayed using tqdm progress bars.

---

## 📊 Results

The model learns meaningful temporal features from video sequences.  
Training curves show convergence behavior and validation performance.

Final accuracy depends on:
- Number of pitch classes
- Dataset balance
- Number of training samples

---

## 🔍 Key Challenges

- Video loading and preprocessing overhead
- Class imbalance in pitch categories
- Temporal modeling sensitivity to frame sampling
- GPU memory constraints for video batches

---

## 🔮 Future Improvements

- ✅ Class balancing and data augmentation  
- ⏳ Try 3D CNNs (C3D / I3D) instead of CNN + LSTM  
- ⏳ Optical flow features for motion modeling  
- ⏳ Transformer-based video models (TimeSformer, ViViT)  
- ⏳ Hyperparameter tuning and cross-validation  

---

## 🧑‍💻 Tech Stack

- Python  
- PyTorch  
- Torchvision  
- OpenCV  
- NumPy / Pandas  
- Matplotlib / Seaborn  

---

## 📌 Acknowledgements

- MLB YouTube Dataset by Piergiovanni et al.  
- PyTorch and Torchvision pretrained models  

---

## 📬 Contact

If you’re interested in discussing this project or related ML work, feel free to connect via GitHub or LinkedIn.
