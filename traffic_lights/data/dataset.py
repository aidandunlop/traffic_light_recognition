import torch
import glob
import os
import pandas as pd
from PIL import Image

from .constants import CLASS_LABEL_MAP


def createDataFrame(files):
    df = pd.concat((pd.read_csv(f, ";") for f in files))
    return df


# TODO: make transforms default to get_transforms from transforms.py


class LISADataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, bad_images=[]):
        self.root = root
        self.transforms = transforms

        # load all annotation files
        annotationFiles = glob.glob(
            os.path.join(root, "Annotations/Annotations", "**/*BOX.csv"), recursive=True
        )
        self.annotations = createDataFrame(annotationFiles)

        # remove folder names from filename column
        self.annotations["Filename"] = (
            self.annotations["Filename"].str.split("/").str[-1]
        )
        # only load images with associated annotations (i.e. traffic lights).
        # Pytorch's API doesn't accept negative training images.
        self.imgs = self.annotations["Filename"].unique()
        # filter out bad images if there are any
        self.imgs = [img for img in self.imgs if img not in bad_images]
        self.annotations = self.annotations[
            ~self.annotations["Filename"].isin(bad_images)
        ]

        self.annotations_groupby_frame = self.annotations.groupby("Filename")

    def getBoxAnnotationsForFrame(self, frame_id):
        annotations = self.annotations_groupby_frame.get_group(frame_id)
        return annotations[
            [
                "Filename",
                "Annotation tag",
                "Upper left corner X",
                "Upper left corner Y",
                "Lower right corner X",
                "Lower right corner Y",
            ]
        ]

    def __getitem__(self, idx):
        # load image at that index
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        annotation = self.getBoxAnnotationsForFrame(self.imgs[idx])
        # get bounding box coordinates for each traffic light
        num_objs = len(annotation)
        boxes = []
        labels = []
        for _, row in annotation.iterrows():
            xmin = row["Upper left corner X"]
            xmax = row["Lower right corner X"]
            ymin = row["Upper left corner Y"]
            ymax = row["Lower right corner Y"]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_LABEL_MAP[row["Annotation tag"]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        # calculate the area for the bounding boxes
        area = (
            (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            if num_objs > 0
            else 0
        )
        area = torch.as_tensor(area, dtype=torch.float32)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "area": area,
            "boxes": boxes,
            "image_id": image_id,
            "iscrowd": iscrowd,
            "labels": labels,
        }
        img_name = self.imgs[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target, img_name

    def __len__(self):
        return len(self.imgs)

    def get_height_and_width(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        width, height = Image.open(img_path).size
        return height, width
