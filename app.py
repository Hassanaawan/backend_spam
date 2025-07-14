import sys
import types
import numpy as np
import tensorflow as tf
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ Added CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🔧 Patch for legacy Keras tokenizer
legacy_module = types.ModuleType("legacy")
legacy_preprocessing_module = types.ModuleType("legacy.preprocessing")
legacy_text_module = types.SimpleNamespace()

from tensorflow.keras.preprocessing.text import Tokenizer
legacy_text_module.Tokenizer = Tokenizer
legacy_preprocessing_module.text = legacy_text_module
legacy_module.preprocessing = legacy_preprocessing_module

sys.modules["keras.src.legacy"] = legacy_module
sys.modules["keras.src.legacy.preprocessing"] = legacy_preprocessing_module
sys.modules["keras.src.legacy.preprocessing.text"] = legacy_text_module

# 🛠 Patch for numpy._core for older pickles
import numpy.core as np_core
sys.modules['numpy._core'] = np_core

# 🧠 Load TFLite model
interpreter = tf.lite.Interpreter(model_path="Model.tflite")
interpreter.allocate_tensors()

# 🔠 Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# 🏷️ Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 📐 Model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
maxlen = input_details[0]['shape'][1]

# 🚀 Flask app
app = Flask(__name__)
CORS(app)  # ✅ Enable CORS for frontend communication


# 🔍 Preprocess input text
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    print(f"🔹 Tokenized sequence: {sequence}")  # DEBUG
    padded = pad_sequences(sequence, maxlen=maxlen, padding='post')
    print(f"🔹 Padded sequence: {padded}")       # DEBUG
    return np.array(padded, dtype=np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        print(f"📥 Input Text: {text}")  # DEBUG

        input_tensor = preprocess_text(text)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        print(f"🔸 Raw Model Output: {output}")  # DEBUG

        # Check output shape and predict accordingly
        if output.shape[-1] == 1:
            prob = output[0][0]
            prediction_idx = int(prob > 0.5)
            confidence = float(prob) if prediction_idx == 1 else float(1 - prob)
        else:
            prediction_idx = np.argmax(output, axis=-1)[0]
            confidence = float(np.max(output))

        print(f"🧾 Prediction Index: {prediction_idx}")  # DEBUG

        classes = label_encoder.classes_
        print(f"🗂️ Classes: {classes}")  # DEBUG

        predicted_label = label_encoder.inverse_transform([prediction_idx])[0]
        print(f"✅ Predicted Label: {predicted_label}")  # DEBUG

        return jsonify({
            "text": text,
            "prediction": predicted_label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ▶️ Run the app
if __name__ == "__main__":
    app.run(debug=False)
