import os
import argparse
from traffic_lights.training.pipeline import training_pipeline
from traffic_lights.inference.predict import predict
import warnings

parser = argparse.ArgumentParser(
    description="Traffic Light Recognition using the LISA dataset"
)
parser.add_argument(
    "train_or_predict",
    help="train or predict",
    choices=("train", "predict"),
    nargs="?",
    default="predict",
)
parser.add_argument(
    "--dataset", "-d", default="lisa-traffic-light-dataset", help="path to dataset",
)
parser.add_argument(
    "--model", "-m", default="model.pth", help="path to model",
)
parser.add_argument(
    "--image", "-i", default="example.jpg", help="path to image for prediction",
)

args = parser.parse_args()
should_train = args.train_or_predict == "train"
dataset = args.dataset
model = args.model
image = args.image

dataset_path = os.path.abspath(dataset)

warnings.filterwarnings("ignore")

if should_train:
    print("Dataset:", dataset)
    training_pipeline(dataset_path)
else:
    print("Model:", model)
    print("Image:", image)
    predict(image, model)
