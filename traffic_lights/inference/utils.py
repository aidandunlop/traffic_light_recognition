from PIL import ImageFont, ImageDraw

colourMap = {
    "go": (0, 255, 0),
    "goForward": (0, 255, 0),
    "goLeft": (0, 255, 0),
    "warning": (255, 127, 0),
    "warningLeft": (255, 127, 0),
    "stop": (255, 0, 0),
    "stopLeft": (255, 0, 0),
}


def threshold_scores(prediction, threshold):
    thresholded_prediction = {
        key: [
            element
            for index, element in enumerate(value)
            if prediction["confidence_scores"][index] > threshold
        ]
        for key, value in prediction.items()
    }
    return thresholded_prediction


def plot_prediction(
    prediction, image, output_file, text_size=20, rect_th=1,
):
    boxes = prediction["boxes"]
    labels = prediction["labels"]

    font = ImageFont.truetype("aller-bold.ttf", text_size)
    for box, classname in zip(boxes, labels):
        box_to_ints = list(map(int, box))
        min_coord = (box_to_ints[0], box_to_ints[1])
        max_coord = (box_to_ints[2], box_to_ints[3])
        light_color = colourMap[classname]
        text_location = (min_coord[0], min_coord[1] - 30)

        draw = ImageDraw.Draw(image)
        draw.rectangle((min_coord, max_coord), outline=light_color, width=rect_th)
        draw.text(text_location, classname, fill=light_color, font=font)
    if output_file:
        image.save(output_file)
    else:
        image.show()
