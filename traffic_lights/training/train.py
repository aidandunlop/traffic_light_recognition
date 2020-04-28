import torch
import numpy as np

from traffic_lights.data.constants import CLASS_LABEL_MAP
from traffic_lights.lib.engine import train_one_epoch
from .model import get_model, evaluate

# TODO: add eval boolean flag, which reports eval stuff if wanted
# TODO: look at amount of parameters y so much


def train(
    parameterization,
    num_classes,
    device,
    data_loader_train,
    data_loader_val,
    validation_ground_truth,
):
    model = get_model(num_classes)
    model.to(device)
    epochs = parameterization["num_epochs"]
    learning_rate = parameterization["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Using lr={} and num_epochs={}".format(learning_rate, epochs))
    for epoch in range(epochs):
        train_one_epoch(
            model, optimizer, data_loader_train, device, epoch, print_freq=1000
        )

    evaluation = evaluate(model, data_loader_val, device, validation_ground_truth)
    precisions = [
        0 if np.isnan(metric["AP"]) else metric["AP"] for metric in evaluation
    ]
    mean_average_precision = np.sum(precisions) / len(CLASS_LABEL_MAP)
    print("mAP:", mean_average_precision)
    torch.save(model, "model_lr_{}_epochs_{}.pth".format(learning_rate, epochs))
    return mean_average_precision
