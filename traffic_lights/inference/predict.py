import torch
import io
from PIL import Image
from traffic_lights.data.constants import REVERSED_CLASS_LABEL_MAP

from .utils import threshold_scores, plot_prediction
from traffic_lights.data.transforms import get_transform


def load_model(model_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    model.to(device)
    return (model, device)


def predict_from_path(stored_image, model, device, threshold=None, output_file=None):
    image = Image.open(stored_image)

    return predict(image, model, device, threshold, output_file)


def predict_from_bytes(stored_image, model, device):
    image = Image.open(io.BytesIO(stored_image))
    return predict(image, model, device)


def predict(image, model, device, threshold=None, output_file=None):
    transformed_image = get_transform()(image.convert("RGB"))
    model_output = model([transformed_image.to(device)])
    prediction = {
        "boxes": model_output[0]["boxes"].tolist(),
        "confidence_scores": model_output[0]["scores"].tolist(),
        "labels": [
            REVERSED_CLASS_LABEL_MAP[label]
            for label in model_output[0]["labels"].tolist()
        ],
    }
    if threshold:
        prediction = threshold_scores(prediction, threshold)
    if output_file:
        plot_prediction(prediction, image, output_file)
    return prediction
