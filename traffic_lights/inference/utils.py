from traffic_lights.data.constants import REVERSED_CLASS_LABEL_MAP
from PIL import ImageFont, ImageDraw
from torchvision import transforms

colourMap = {
    "go": (0, 255, 0),
    "goForward": (0, 255, 0),
    "goLeft": (0, 255, 0),
    "warning": (255, 127, 0),
    "warningLeft": (255, 127, 0),
    "stop": (255, 0, 0),
    "stopLeft": (255, 0, 0),
}


def get_prediction(img, model, threshold, device):
    pred = model([img.to(device)])
    pred_class = [
        REVERSED_CLASS_LABEL_MAP[i] for i in list(pred[0]["labels"].cpu().numpy())
    ]
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])]
        for i in list(pred[0]["boxes"].cpu().detach().numpy())
    ]
    pred_score = list(pred[0]["scores"].cpu().detach().numpy())
    print("predicted labels:", pred_class)
    print("predicted scores:", pred[0]["scores"].tolist())
    accepted_scores = [pred_score.index(x) for x in pred_score if x > threshold]
    if len(accepted_scores) <= 0:
        return ([], [])

    pred_t = accepted_scores[-1]
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_class = pred_class[: pred_t + 1]
    return pred_boxes, pred_class


def plot_detection(
    image, model, device, threshold=0.5, rect_th=1, text_size=20, text_th=2
):
    boxes, pred_cls = get_prediction(image, model, threshold, device)
    trans = transforms.ToPILImage()

    im = trans(image)
    font = ImageFont.truetype("aller-bold.ttf", text_size)

    for box, classname in zip(boxes, pred_cls):
        min_coord = tuple(map(int, box[0]))
        max_coord = tuple(map(int, box[1]))
        light_color = colourMap[classname]
        text_location = (min_coord[0], min_coord[1] - 30)

        draw = ImageDraw.Draw(im)
        draw.rectangle((min_coord, max_coord), outline=light_color, width=rect_th)
        draw.text(text_location, classname, fill=light_color, font=font)

    im.save("detection.png")
    im.show()
