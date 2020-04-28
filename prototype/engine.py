import math
import sys
import time
import torch

import engine_utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = engine_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', engine_utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # skip images with nothing in
        targets = [item for item in targets if not item['boxes'].equal(
            torch.tensor([]).to(device))]
        images = images[:len(targets)]

        if len(targets) > 0:
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = engine_utils.reduce_dict(loss_dict)
            # losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            # loss_value = losses_reduced.item()

            # if not math.isfinite(loss_value):
            #     print("Loss is {}, stopping training".format(loss_value))
            #     print(loss_dict_reduced)
            #     sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
