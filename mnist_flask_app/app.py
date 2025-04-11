from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from keras.models import load_model

app = Flask(__name__)
model = load_model("D:\Projects\MNIST Image Classification(Hand Written Digits)\mnist_flask_app\model\mnist_model.h5")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data_url = request.json['image']
    encoded_data = data_url.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (28, 28))
    img = 255 - img  # invert
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    return jsonify({'prediction': int(digit)})

if __name__ == "__main__":
    app.run(debug=True)
