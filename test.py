# import sys
# import numpy.core as np_core
# sys.modules['numpy._core'] = np_core  # ✅ patch

# import pickle

# with open("label_encoder.pkl", "rb") as f:
#     label_encoder = pickle.load(f)

# print("✅ LabelEncoder loaded successfully!")
# print("Classes:", label_encoder.classes_)
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🚀 Rebuild LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(["ham", "spam"])

# 🚀 Rebuild Tokenizer (dummy fit to init it)
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(["dummy"])

# 🔧 Load TFLite model
interpreter = tf.lite.Interpreter(model_path="Model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
maxlen = input_details[0]['shape'][1]

# 🎯 Preprocess
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=maxlen, padding='post')
    return np.array(padded, dtype=np.float32)

# 🌐 Flask API
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        input_tensor = preprocess_text(text)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        prediction_idx = np.argmax(output, axis=-1)[0]
        predicted_label = label_encoder.inverse_transform([prediction_idx])[0]

        return jsonify({
            "text": text,
            "prediction": predicted_label,
            "confidence": float(np.max(output))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
