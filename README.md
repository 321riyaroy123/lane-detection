# Lane Detection Project 🚗

This project implements **lane detection** on road images and videos using OpenCV and Computer Vision techniques.  
It can process both static images and dynamic video streams to detect lane markings, which is useful for autonomous driving and driver-assistance systems.

---

## 📂 Project Structure
```
lanedetectiondataset/   # Dataset (not included in repo, provide external link)
testimage/              # Test images (sample included for quick testing)
testvideos/             # Test videos (sample included for quick testing)
results/                # Output results (generated after running the code)
code/                   # Python scripts and Jupyter notebooks
requirements.txt        # Python dependencies
README.md               # Project documentation (this file)
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/321riyaroy123/lane-detection.git
cd lane-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run lane detection on a sample image:
```bash
python code/lane_detection.py --input testimage/sample.jpg
```

Run lane detection on a sample video:
```bash
python code/lane_detection.py --input testvideos/sample.mp4
```

The results will be saved in the `results/` folder.

---

## 📊 Dataset

Due to size limitations, the dataset is not included in the repository.  
You can download it from: [Google Drive / Kaggle link here].

Make sure to place the dataset inside the `lanedetectiondataset/` folder.

---

## 🔑 Features

- Lane detection using edge detection and Hough Transform.  
- Works on both images and videos.  
- Region of interest (ROI) masking for better lane focus.  
- Can be extended with deep learning models for advanced lane detection.  

---

## 🎥 Example Output

*(Add a GIF or image from the results folder here to showcase your output)*

---

## 🛠️ Technologies Used

- Python 3.x  
- OpenCV  
- NumPy  
- Matplotlib (for visualization)  
- (Optional) TensorFlow / PyTorch for deep learning models  

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👩‍💻 Author

Developed by **Riya Roy (321riyaroy123)**  
Feel free to contribute or raise issues!
