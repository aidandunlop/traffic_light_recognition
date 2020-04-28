from traffic_lights.lib.utils import BBType, BBFormat, CoordinatesType
from traffic_lights.lib.BoundingBoxes import BoundingBoxes
from traffic_lights.lib.BoundingBox import BoundingBox


def create_bounding_boxes(data_loader, bb_type, model, device, ground_truth_bb=None):
    print("Creating bounding boxes...")
    is_detection = bb_type == BBType.Detected
    if is_detection and ground_truth_bb is not None:
        model.eval()
        bounding_boxes = ground_truth_bb
    elif is_detection:
        raise Exception("please provide ground truth bounding boxes.")
    else:
        bounding_boxes = BoundingBoxes()

    for images, targets, image_names in data_loader:
        if is_detection:
            images = list(img.to(device) for img in images)
            annotations = model(images)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        else:
            annotations = [{k: v.to(device) for k, v in t.items()} for t in targets]

        for annotation, image_name in zip(annotations, image_names):
            boxes = annotation["boxes"].tolist()
            labels = annotation["labels"].tolist()
            scores = annotation["scores"].tolist() if "scores" in annotation else None
            items = zip(boxes, labels, scores) if is_detection else zip(boxes, labels)

            for item in items:
                if is_detection:
                    box, label, score = item
                else:
                    box, label = item

                x1, y1, x2, y2 = box
                bounding_box_args = {
                    "imageName": image_name,
                    "classId": label,
                    "x": x1,
                    "y": y1,
                    "w": x2,
                    "h": y2,
                    "typeCoordinates": CoordinatesType.Absolute,
                    "bbType": bb_type,
                    "format": BBFormat.XYX2Y2,
                }
                if is_detection:
                    bounding_box_args["classConfidence"] = score
                bb = BoundingBox(**bounding_box_args)
                bounding_boxes.addBoundingBox(bb)

    return bounding_boxes
