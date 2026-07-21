# Pill Identification Flask App

## Folder structure
```
pill_flask_app/
├── app.py
├── model.pkl
├── requirements.txt
├── templates/
│   └── index.html
```

## Setup

1. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the app:
   ```
   python app.py
   ```

4. Open your browser at `http://127.0.0.1:5000/` and upload a pill image.

## Notes

- The model expects images resized to 224x224 and preprocessed with
  `tensorflow.keras.applications.efficientnet.preprocess_input`, matching how
  it was trained (EfficientNetB0 backbone).
- Class labels: Alaxan, Bactidol, Bioflu, Biogesic, DayZinc, Decolgen,
  Fish Oil, Kremil S, Medicol, Neozep.
- If `model.pkl` fails to load (e.g. due to a TensorFlow version mismatch
  between Colab and your local machine), re-save the model from the notebook
  using `model.save("model.h5")` and load it in `app.py` with
  `tf.keras.models.load_model("model.h5")` instead of pickle.
