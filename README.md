# American Sign Language (ASL) to Text and Speech Conversion

## Project Overview

This project focuses on building a real-time system that recognizes **American Sign Language (ASL)** fingerspelling gestures and converts them into **text** and **speech** outputs.  
It aims to bridge the communication gap for the **Deaf and Mute (D&M)** community by enabling seamless interaction with those who do not understand sign language.

The system uses **MediaPipe** for hand landmark detection, **OpenCV** for preprocessing, and a **Convolutional Neural Network (CNN)** model for gesture classification. Text outputs are further converted to speech using **pyttsx3**.

---

## Objectives

- Recognize **static hand gestures** representing the 26 alphabets (A-Z) in ASL.
- Translate recognized gestures into **text**.
- Provide a **text-to-speech (TTS)** functionality to vocalize the text.
- Build an **accessible, real-time Human-Computer Interface (HCI)** for D&M users.

---

## System Architecture

1. **Data Acquisition:**  
   Capture hand gesture images using a regular webcam.

2. **Preprocessing & Feature Extraction:**  
   - Detect hand landmarks using **MediaPipe**.  
   - Plot the landmarks on a **plain white background** to simplify input.

3. **Gesture Classification:**  
   - A trained **CNN** model classifies gestures corresponding to alphabets.

4. **Text and Speech Conversion:**  
   - Recognized alphabets form words.  
   - The words can be **spoken aloud** using a TTS engine.

---

## Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe** (for hand landmark detection)
- **TensorFlow / Keras** (for CNN model training)
- **pyttsx3** (for Text-to-Speech)

---

## Dataset Download

This dataset contains images representing hand signs for the 26 English alphabets (A–Z). 

🔗 [Click here to download the dataset from Google Drive](https://drive.google.com/file/d/13hnnJ35bWtV1B8gOzYr0iasq3SL2KJJl/view?usp=sharing)

---

## Results

- Achieved an overall accuracy of **99%** under ideal conditions.
- Real-time gesture recognition and text generation.
- Added options for **Word Suggestions**, **Speak**, and **Clear** functionalities.
- Confusion matrix and per-class F1 scores show high reliability across all alphabets.

---

## Features

- Real-time **alphabet recognition** (A-Z).
- **Word formation** by sequential recognition.
- **Speech output** for the formed text.
- **Buttons** for easy interaction:  
  - **Speak**: Vocalizes the text.  
  - **Clear**: Resets the text field.

---

## Dataset

- **Custom Dataset** collected using webcam.
- **Landmark-based** (coordinates of key hand points) representation.
- 50–60 samples per alphabet collected under various lighting and hand orientations.
- Preprocessed into binary images for efficient training.

---

## How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/harinii19/ASL-to-Text-Speech-Conversion.git
   cd ASL-to-Text-Speech-Conversion
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python final_pred.py
   ```

4. Perform hand gestures in front of your webcam to see real-time recognition!

---

## Future Enhancements

- Recognize **numeric gestures** and **special symbols**.
- Include **common words** and **phrases** for faster communication.
- Extend to **dynamic gesture** recognition (for complex ASL signs).
- Improve generalization across **different hand sizes, skin tones**, and **backgrounds**.

---

## Contributors

- **Aishwarya S**
- **Harini Natarajan**
- **Priyanka Akilan**
- **Subhasri M**
- **Yuktanidhi C**

---

## References

- [Conversion of Sign Language to Text and Speech using ML Techniques (ResearchGate)](https://www.researchgate.net/publication/335433017_Conversion_of_Sign_Language_To_Text_And_Speech_Using_Machine_Learning_Techniques)
- [Translation of Sign Language Fingerspelling to Text using Image Processing (IJCA)](https://research.ijcaonline.org/volume77/number11/pxc3891313.pdf)
- [An Improved Hand Gesture Recognition Algorithm (IOP Science)](https://iopscience.iop.org/article/10.1088/1757-899X/1116/1/012115)

