"""---------------- REMEDIES ----------------"""

REMEDIES = {
    "Potato___healthy": {
        "description": "Healthy potato plant",
        "remedies": [
            "No action needed",
            "Maintain good growing conditions"
        ]
    },
    "Potato___Early_blight": {
        "description": "Fungal disease causing dark spots on leaves",
        "remedies": [
            "Remove and destroy infected leaves",
            "Apply copper-based fungicides",
            "Practice crop rotation"
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Fungal disease causing small dark spots with light centers",
        "remedies": [
            "Remove affected leaves",
            "Use fungicides like chlorothalonil",
            "Avoid overhead watering"
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Viral mosaic-pattern disease",
        "remedies": [
            "Destroy infected plants",
            "Control aphids",
            "Use resistant varieties"
        ]
    },
    "Pepper__bell___healthy": {
        "description": "Healthy bell pepper plant",
        "remedies": ["No action needed"]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Spider mites causing yellowing and webbing",
        "remedies": [
            "Use neem oil",
            "Increase humidity",
            "Introduce predatory mites"
        ]
    },
    "Tomato__Target_Spot": {
        "description": "Brown spots with yellow halos",
        "remedies": [
            "Remove infected leaves",
            "Apply copper fungicides",
            "Improve ventilation"
        ]
    },
    "Tomato_healthy": {
        "description": "Healthy tomato plant",
        "remedies": ["No action needed"]
    },
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial disease with water-soaked spots",
        "remedies": [
            "Remove infected leaves",
            "Use copper sprays",
            "Choose resistant varieties"
        ]
    },
    "Tomato_Late_blight": {
        "description": "Severe fungal disease with brown lesions",
        "remedies": [
            "Destroy infected plants",
            "Spray chlorothalonil fungicide",
            "Keep area dry"
        ]
    },
    "Potato___Late_blight": {
        "description": "Late blight affecting potato leaves and tubers",
        "remedies": [
            "Use resistant varieties",
            "Apply fungicides",
            "Avoid excess moisture"
        ]
    },
    "Tomato_Early_blight": {
        "description": "Dark concentric rings on leaves",
        "remedies": [
            "Use copper fungicides",
            "Remove affected leaves",
            "Mulch to reduce soil splash"
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Virus causing curling and yellowing",
        "remedies": [
            "Destroy infected plants",
            "Control whiteflies",
            "Use resistant varieties"
        ]
    },
    "Tomato_Bacterial_spot": {
        "description": "Water-soaked bacterial spots",
        "remedies": [
            "Remove infected plants",
            "Apply copper spray",
            "Use clean seeds"
        ]
    },
    "Tomato_Leaf_Mold": {
        "description": "Yellowing + mold on leaves",
        "remedies": [
            "Improve ventilation",
            "Apply copper spray",
            "Avoid overhead watering"
        ]
    }
}

"""--------------- PREDICTION ---------------"""

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("/content/drive/MyDrive/best_model.h5")

# Load class indices for mapping
class_indices = np.load('class_indices.npy', allow_pickle=True).item()
idx_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    # Preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions)

    class_name = idx_to_class[class_id]

    # Remedies fallback
    info = REMEDIES.get(class_name, {
        "description": "No description available",
        "remedies": ["Consult an agricultural specialist"]
    })

    # Show image + prediction
    plt.imshow(img)
    plt.title(f"{class_name} ({confidence:.2%})")
    plt.axis("off")
    plt.show()

    print("\nDisease:", class_name)
    print("Description:", info["description"])
    print("\nRecommended Remedies:")
    for i, r in enumerate(info["remedies"], 1):
        print(f"{i}. {r}")