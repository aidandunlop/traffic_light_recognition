import boto3
import os
import warnings
from flask import Flask, jsonify, request
from traffic_lights.inference.predict import load_model, predict_from_bytes

app = Flask(__name__)


def download_model():
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
        file = s3.download_file(os.environ["AWS_BUCKET"], "model.pth", "api/model.pth")
    except boto3.exceptions.ResourceNotExistsError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise

    return file


@app.before_first_request
def before_first_request():
    download_model()
    warnings.filterwarnings("ignore")
    model_path = "./model.pth"
    global model, device
    model, device = load_model(model_path)


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
