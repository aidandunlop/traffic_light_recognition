import warnings
from flask import Flask, jsonify, request
from traffic_lights.inference.predict import load_model, predict_from_bytes

app = Flask(__name__)

warnings.filterwarnings("ignore")
model_path = "./model.pth"
model, device = load_model(model_path)


@app.route("/predict", methods=["POST"])
def predict_image():
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
        prediction = predict_from_bytes(img_bytes, model, device)
        result = jsonify(prediction)
        result.status_code = 200
    return result
