import unittest
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader
from test.utils import create_mock_dataset, create_mock_data
from traffic_lights.data.loading import split_data


class TestLoading(unittest.TestCase):
    @classmethod
    @patch(
        "traffic_lights.data.dataset.pd.concat",
        MagicMock(return_value=create_mock_data()),
    )
    def setUpClass(self):
        self.dataset = create_mock_dataset()
        self.data_loaders = split_data(self.dataset)

    def test_data_loaders_are_all_torch_dataloaders(self):
        for loader in self.data_loaders:
            self.assertIsInstance(loader, DataLoader)

    def test_data_loaders_length(self):
        (
            training_dataloader,
            validation_dataloader,
            testing_dataloader,
        ) = self.data_loaders
        self.assertEqual(len(training_dataloader), 1)
        self.assertEqual(len(validation_dataloader), 1)
        self.assertEqual(len(testing_dataloader), 1)


if __name__ == "__main__":
    unittest.main()
