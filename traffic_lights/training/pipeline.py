import torch
import numpy as np
from time import perf_counter
from ax.service.managed_loop import optimize

from traffic_lights.data.constants import bad_images, CLASS_LABEL_MAP
from traffic_lights.data.dataset import LISADataset
from traffic_lights.data.transforms import get_transform
from traffic_lights.data.loading import split_data
from traffic_lights.lib.utils import BBType
from .utils import create_bounding_boxes
from .train import train
from .model import evaluate

# TODO: make params config configurable


def training_pipeline(dataset_path):
    start_training = perf_counter()
    # train on the GPU or on the CPU if a GPU is not available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Training on the", device)

    # Load the dataset. Remove any unwanted images
    full_dataset = LISADataset(
        dataset_path, transforms=get_transform(), bad_images=bad_images
    )

    # split data into training, testing and validation sets
    data_loader_train, data_loader_test, data_loader_val = split_data(full_dataset)

    validation_ground_truth = create_bounding_boxes(
        data_loader_val, BBType.GroundTruth, model=None, device=device
    )

    num_classes = len(CLASS_LABEL_MAP) + 1  # add 1 for background class

    # hyperparameter tuning
    best_parameters, values, experiment, ax_model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.01], "log_scale": True},
            {"name": "num_epochs", "type": "choice", "values": list(range(1, 3))},
        ],
        evaluation_function=lambda params: train(
            params,
            num_classes,
            device,
            data_loader_train,
            data_loader_val,
            validation_ground_truth,
        ),
        objective_name="mean_average_precision",
        total_trials=1,
    )
    end_training = perf_counter()
    training_time = end_training - start_training
    print("Training took", training_time / 60.0, "minutes")

    # evaluation
    start_eval = perf_counter()
    path_to_best_model = "model_lr_{}_epochs_{}.pth".format(
        best_parameters["lr"], best_parameters["num_epochs"]
    )
    print("Best model saved at", path_to_best_model)
    model = torch.load(path_to_best_model, map_location=torch.device(device))

    testing_ground_truth = create_bounding_boxes(
        data_loader_test, BBType.GroundTruth, model, device
    )

    print("getting evaluation scores...")
    evaluation = evaluate(model, data_loader_test, device, testing_ground_truth)

    # TODO: move map calculation into evaluation, just return m_a_p
    precisions = [
        0 if np.isnan(metric["AP"]) else metric["AP"] for metric in evaluation
    ]
    mean_average_precision = np.sum(precisions) / len(CLASS_LABEL_MAP)
    print("Mean average precision:", mean_average_precision)

    end_eval = perf_counter()
    eval_time = end_eval - start_eval
    print("Evaluation took", eval_time / 60.0, "minutes")
