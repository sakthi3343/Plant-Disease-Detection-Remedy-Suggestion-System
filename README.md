🌿 Plant Disease Detection Using Deep Learning

This project provides an AI-powered system that identifies plant diseases from leaf images using a Convolutional Neural Network (CNN).
It includes a trained model, prediction scripts, remedies, and a simple web application for easy usage.

🔥 Key Features

🌱 Detects plant diseases from uploaded leaf images

📊 Displays prediction confidence

💡 Shows disease details and suggested treatments

⚡ Built with a fast and lightweight Streamlit interface

🖼 Includes sample images and UI screenshots

🔧 Installation & Setup
1️⃣ Clone this repository
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME

2️⃣ Install required libraries
pip install -r requirements.txt

3️⃣ Download the trained CNN model

Download the .h5 model file from the link below:

🔗 Model Download:
https://drive.google.com/file/d/190XKlIX47r497C2W54c1Env5c-WAH8zu/view?usp=sharing

After downloading, move the file to the project directory and rename it:

best_model.h5

▶️ Launching the Streamlit App

Run the following command to start the web interface:

streamlit run app.py


Once the app opens, you can upload a leaf image to view:

Predicted disease

Confidence percentage

Disease description

Recommended remedies

🧪 CLI Example

You can also test the model directly using the prediction script:

python predict.py sample_images/sample1.jpg


Sample Output:

Predicted: Tomato___Late_blight (92.17%)

🖼 Application Screenshots
🏠 Home Screen
<img src="screenshots/Home.png" width="600">
📤 Image Upload Page
<img src="screenshots/Upload.png" width="600">
📊 Prediction Result
<img src="screenshots/Output.png" width="600">
📁 Repository Overview
📦 Plant Disease Detector
│── app.py
│── predict.py
│── requirements.txt
│── class_indices.npy
│── remedies.json
│── README.md
│── sample_images/
│── screenshots/

🧠 Model Information

Framework: TensorFlow / Keras

Input resolution: 224 × 224

Dataset: PlantVillage

🙏 Credits

This project was developed for academic learning and utilizes publicly available datasets and open-source deep learning tools.
