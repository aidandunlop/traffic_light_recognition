import pandas as pd
from PIL import Image
from traffic_lights.data.dataset import LISADataset
from traffic_lights.data.transforms import get_transform


def create_mock_image(as_tensor=False):
    image = Image.new("RGB", size=(50, 50), color=(155, 0, 0))
    if as_tensor:
        transform = get_transform()
        return transform(image)  # convert this to tensor
    return image


def create_mock_data():
    df = pd.DataFrame(
        {
            "Filename": [
                "dayClip2/dayClip2--0001.jpg",
                "dayClip2/dayClip2--0001.jpg",
                "dayClip2/dayClip2--0001.jpg",
                "nightClip2/nightClip2--0001.jpg",
                "nightClip2/nightClip2--0001.jpg",
                "nightClip2/nightClip2--0001.jpg",
                "nightClip2/nightClip2--0002.jpg",
                "nightClip2/nightClip2--0002.jpg",
            ],
            "Annotation tag": ["go", "go", "go", "go", "go", "go", "go", "go"],
            "Upper left corner X": [100, 200, 300, 100, 200, 300, 100, 200],
            "Upper left corner Y": [100, 200, 300, 100, 200, 300, 100, 200],
            "Lower right corner X": [10, 120, 230, 10, 120, 230, 10, 120],
            "Lower right corner Y": [10, 120, 230, 10, 120, 230, 10, 120],
            "Origin file": [
                "dayClip2.mp4",
                "dayClip2.mp4",
                "dayClip2.mp4",
                "nightClip2.mp4",
                "nightClip2.mp4",
                "nightClip2.mp4",
                "nightClip2.mp4",
                "nightClip2.mp4",
            ],
            "Origin frame number": [1, 1, 1, 1, 1, 1, 2, 2],
            "Origin track": [1, 1, 1, 1, 1, 1, 2, 2],
            "Origin track frame number": [1, 1, 1, 1, 1, 1, 2, 2],
        }
    )
    return df


def create_mock_dataset():
    return LISADataset("mock_path", transforms=get_transform())
