import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import torch
from torch.utils.data import Dataset
from traffic_lights.data.dataset import LISADataset
from traffic_lights.data.transforms import get_transform
from test.utils import create_mock_data, create_mock_image


@patch(
    "traffic_lights.data.dataset.Image.open",
    MagicMock(return_value=create_mock_image()),
)
class TestLISADataset(unittest.TestCase):
    @classmethod
    @patch(
        "traffic_lights.data.dataset.pd.concat",
        MagicMock(return_value=create_mock_data()),
    )
    def setUpClass(self):
        self.dataset = LISADataset("mock_path", transforms=get_transform())

    def test_instance_of_torch_dataset(self):
        self.assertIsInstance(self.dataset, Dataset)

    def test_annotation_dataframe_columns(self):
        annotion_dataframe = self.dataset.annotations
        self.assertIsInstance(annotion_dataframe, pd.DataFrame)
        expected_columns = [
            "Filename",
            "Annotation tag",
            "Upper left corner X",
            "Upper left corner Y",
            "Lower right corner X",
            "Lower right corner Y",
            "Origin file",
            "Origin frame number",
            "Origin track",
            "Origin track frame number",
        ]
        self.assertListEqual(list(annotion_dataframe), expected_columns)

    def test_filename_stripping(self):
        filename = self.dataset.annotations["Filename"].values[0]
        self.assertEqual(filename, "dayClip2--0001.jpg")

    def test_length_getter(self):
        self.assertEqual(self.dataset.__len__(), 3)

    def test_get_height_and_width(self):
        self.assertEqual(self.dataset.get_height_and_width(0), (50, 50))

    def test_get_item(self):
        item = self.dataset.__getitem__(0)
        expected = {
            "area": torch.tensor([8100, 6400, 4900]),
            "boxes": torch.tensor(
                [
                    [100.0, 100.0, 10.0, 10.0],
                    [200.0, 200.0, 120.0, 120.0],
                    [300.0, 300.0, 230.0, 230.0],
                ]
            ),
            "image_id": torch.tensor([0]),
            "iscrowd": torch.tensor([0]),
            "labels": torch.tensor([1]),
        }
        self.assertTrue(torch.all(torch.eq(item[0], create_mock_image(as_tensor=True))))

        for key, value in item[1].items():
            self.assertTrue(
                torch.all(torch.eq(value, expected[key])), msg=f"{key} does not match",
            )

    @patch(
        "traffic_lights.data.dataset.pd.concat",
        MagicMock(return_value=create_mock_data()),
    )
    def test_removes_bad_images(self):
        smaller_dataset = LISADataset(
            "mock_path", transforms=get_transform(), bad_images=["nightClip2--0001.jpg"]
        )
        self.assertEqual(smaller_dataset.__len__(), 2)


if __name__ == "__main__":
    unittest.main()
