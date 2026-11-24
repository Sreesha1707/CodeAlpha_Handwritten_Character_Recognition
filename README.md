# Handwritten Digit and Character Recognition

## Project Overview
This project is an AI-powered Handwritten Digit and Character Recognition system developed using Python, TensorFlow, and Streamlit.

It uses a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) and uppercase English letters (A–Z) from uploaded images.

## Features
- Recognizes handwritten digits from 0 to 9.
- Recognizes uppercase English letters from A to Z.
- Supports image uploads.
- Displays the predicted digit or character.
- Shows the prediction confidence percentage.
- Provides a simple and user-friendly web interface.

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow (PIL)
- Streamlit
- EMNIST Dataset

## Model Details
A Convolutional Neural Network (CNN) is used for image classification.

The model is trained using the EMNIST dataset, selecting 36 classes:
- 10 digit classes (0–9)
- 26 uppercase letter classes (A–Z)

**Reported test accuracy:** 93.15%

## Project Structure
```text
CodeAlpha_Handwritten_Character_Recognition/
│
├── app.py
├── train_character_model.py
├── handwritten_character_model.keras
├── README.md
└── .gitignore
```

## Installation and Setup

1. Clone the repository:
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Open the project folder:
```bash
cd CodeAlpha_Handwritten_Character_Recognition
```

3. Create a virtual environment:
```bash
python -m venv .venv
```

4. Activate the virtual environment on Windows:
```bash
.venv\Scripts\activate
```

5. Install the required libraries:
```bash
pip install tensorflow numpy matplotlib pillow streamlit tensorflow-datasets importlib-resources
```

## Run the Application

Start the Streamlit application using:

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

Upload a handwritten digit or uppercase letter image, select the recognition mode, and view the prediction.

## Project Objective
The objective of this project is to demonstrate how machine learning and deep learning techniques can be used to recognize handwritten digits and letters through image classification.

## Internship
Developed as part of the **CodeAlpha Machine Learning Internship**.
