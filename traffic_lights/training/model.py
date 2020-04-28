import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from traffic_lights.lib.Evaluator import Evaluator
from traffic_lights.lib.utils import BBType
from .utils import create_bounding_boxes


def get_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def evaluate(model, data_loader, device, ground_truth_bb):
    all_bb = create_bounding_boxes(
        data_loader, BBType.Detected, model, device, ground_truth_bb
    )

    evaluator = Evaluator()
    metrics = evaluator.GetPascalVOCMetrics(all_bb)
    print("Average precision per class:")
    for metric in metrics:
        c = metric["class"]
        # Get metric values per each class
        average_precision = metric["AP"]
        # Print AP per class
        print("%s: %f" % (c, average_precision))
    return metrics
