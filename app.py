import pickle
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

# Load model once at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

CLASS_NAMES = [
    "Alaxan", "Bactidol", "Bioflu", "Biogesic", "DayZinc",
    "Decolgen", "Fish Oil", "Kremil S", "Medicol", "Neozep"
]

# General "what it's commonly used for" info shown alongside a prediction.
# This is informational only, not medical advice — always defer to the
# product label, a pharmacist, or a doctor for actual usage/dosage.
MEDICINE_INFO = {
    "Alaxan": {
        "generic_name": "Ibuprofen + Paracetamol",
        "category": "Pain reliever (NSAID + analgesic combo)",
        "used_for": "Headaches, muscle and body pain, dental pain, menstrual cramps, and fever.",
    },
    "Bactidol": {
        "generic_name": "Hexetidine (antiseptic)",
        "category": "Oral antiseptic",
        "used_for": "Sore throat, mouth infections, gum problems, and bad breath (gargle/lozenge).",
    },
    "Bioflu": {
        "generic_name": "Paracetamol + Phenylephrine + Chlorphenamine",
        "category": "Flu and cold relief",
        "used_for": "Flu symptoms such as fever, colds, headache, and body pain.",
    },
    "Biogesic": {
        "generic_name": "Paracetamol",
        "category": "Analgesic / antipyretic",
        "used_for": "Fever and mild to moderate pain such as headache and body pain.",
    },
    "DayZinc": {
        "generic_name": "Zinc",
        "category": "Mineral supplement",
        "used_for": "Supports the immune system and may help shorten the duration of colds.",
    },
    "Decolgen": {
        "generic_name": "Phenylephrine + Paracetamol + Chlorphenamine",
        "category": "Cold and flu relief",
        "used_for": "Colds and flu — nasal congestion, runny nose, fever, and headache.",
    },
    "Fish Oil": {
        "generic_name": "Omega-3 fatty acids (EPA/DHA)",
        "category": "Dietary supplement",
        "used_for": "General heart and cholesterol health support.",
    },
    "Kremil S": {
        "generic_name": "Aluminum hydroxide + Magnesium hydroxide + Simethicone",
        "category": "Antacid",
        "used_for": "Heartburn, acid indigestion, hyperacidity, and bloating.",
    },
    "Medicol": {
        "generic_name": "Ibuprofen",
        "category": "NSAID pain reliever",
        "used_for": "Pain, fever, and inflammation such as headaches and muscle pain.",
    },
    "Neozep": {
        "generic_name": "Phenylephrine + Paracetamol + Chlorphenamine",
        "category": "Cold and flu relief",
        "used_for": "Colds and flu — nasal congestion, sneezing, fever, and headache.",
    },
}

IMG_SIZE = (224, 224)


def prepare_image(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_array = prepare_image(file)
        prediction = model.predict(img_array, verbose=0)
        predicted_class = CLASS_NAMES[int(np.argmax(prediction))]
        confidence = float(np.max(prediction))
        info = MEDICINE_INFO.get(predicted_class, {})

        return jsonify({
            "medicine": predicted_class,
            "confidence": round(confidence * 100, 2),
            "generic_name": info.get("generic_name", ""),
            "category": info.get("category", ""),
            "used_for": info.get("used_for", "")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
