import warnings
from flask import Flask, jsonify, request
from traffic_lights.inference.predict import load_model, predict_from_bytes

TEMP_MODAL_PATH = "/tmp/tlr_model.pth"


app = Flask(__name__)


@app.before_first_request
def before_first_request():
    print("Loading the model...")
    warnings.filterwarnings("ignore")
    global model, device
    model, device = load_model(TEMP_MODAL_PATH)
    print("Model loaded.")


@app.route("/")
def entry():
    return (
        "Traffic Light recognition using Pytorch. POST an image to /predict to try it"
    )


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


if __name__ == "__main__":
    app.run()
