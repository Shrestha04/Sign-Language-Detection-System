## 🤟 Sign-Language-Detection-System
 
This project is a real-time Sign Language Detection System that uses a webcam to recognize American Sign Language (ASL) hand gestures. It employs **MediaPipe** for hand landmark detection and a **Random Forest Classifier** trained on extracted keypoints.

---

## 📌 Features

- Real-time webcam-based gesture recognition.
- Custom dataset collection from scratch.
- Hand landmark extraction using Google MediaPipe.
- Landmark-based gesture classification using Random Forest.
- Model accuracy tracking.
- Modular and extendable codebase.

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe**
- **scikit-learn**
- **NumPy**
- **Matplotlib**
- **Pickle**

---

## 📁 Project Structure

Sign-Language-Detection-System/
- ├── collect_imgs.py  (# Script to collect gesture images from webcam)
- ├── create_dataset.py  (# Extract landmarks from images and create dataset)
- ├── train_classifier.py  (# Train a classifier using the processed dataset)
- ├── inference_classifier.py  (# Real-time gesture recognition using webcam)
- ├── check_accuracy.py  (# Real-time prediction with accuracy calculation)

After running the above code files, following files will be generated:
- ├── data.pickle  (# Pickled file with landmark data and labels)
- ├── model.p  (# Trained machine learning model)
- └── data/  (# Folder containing gesture images (organized by class))

---

## 🧠 How It Works

1. **Image Collection (`collect_imgs.py`)**
   - Collects 300 samples per class using webcam.
   - Prompts the user for each gesture.
   - Saves images in `./data/<class_id>/`.

2. **Dataset Creation (`create_dataset.py`)**
   - Extracts 21 hand landmarks per image using MediaPipe.
   - Normalizes landmark positions.
   - Saves the landmark features and labels into `data.pickle`.

3. **Model Training (`train_classifier.py`)**
   - Loads the dataset.
   - Trains a `RandomForestClassifier`.
   - Evaluates accuracy on a test split.
   - Saves the trained model into `model.p`.

4. **Real-Time Inference (`inference_classifier.py`)**
   - Captures webcam feed.
   - Predicts the gesture from real-time hand landmarks.
   - Displays prediction on screen with bounding box.

5. **Accuracy Evaluation (`check_accuracy.py`)**
   - Compares real-time predictions with ground truth (manually set).
   - Displays and calculates live prediction accuracy.

---

## 🧪 Setup Instructions

1. **Clone the repository**
- git clone https://github.com/Shrestha04/Sign-Language-Detection-System.git
- cd Sign-Language-Detection-System

2. **Install Dependencies**
- pip install opencv-python mediapipe scikit-learn numpy matplotlib

3. **Run Scripts in Order**
- Step 1: Collect Gesture Images - python collect_imgs.py
- Step 2: Create Dataset - python create_dataset.py
- Step 3: Train the Classifier - python train_classifier.py
- Step 4: Run Real-time Inference - python inference_classifier.py
- Optional: Evaluate Accuracy in Real-Time - python check_accuracy.py

--- 

## 🖐️ Label Mapping
- Each class label (0–24) corresponds to a character in the English alphabet:
- 0 = A, 1 = B, 2 = C, ..., 24 = Y, 25 = Z
- Make sure the actual gesture shown matches the assigned class during collection!

---

## 📈 Results

- Model: RandomForestClassifier
- Accuracy (on test set): ~92% 
- Real-time performance: ~30 FPS (depending on system)

---

## 🚀 Future Improvements

- Use a deep learning model like CNN or LSTM for better accuracy.
- Expand to include dynamic signs or phrases.
- Implement gesture-to-text or gesture-to-speech functionality.
- Add a GUI for user interaction.

---

## 📄 License
This project is open-source and available under the MIT License.

---

### Made with ❤️ by Shrestha!





