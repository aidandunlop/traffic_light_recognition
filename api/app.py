import warnings
from flask import Flask, jsonify, request
from traffic_lights.inference.predict import load_model, predict_from_bytes
from .download_model import temp_modal_path

app = Flask(__name__)


@app.before_first_request
def before_first_request():
    print("Loading the model...")
    warnings.filterwarnings("ignore")
    global model, device
    model, device = load_model(temp_modal_path)
    print("Model loaded.")


@app.route("/predict", methods=["POST"])
def predict_image():
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
        if model is None or device is None:
            raise Exception("Model or device not found")
        prediction = predict_from_bytes(img_bytes, model, device)
        result = jsonify(prediction)
        result.status_code = 200
    return result
