import unittest
from mock import MagicMock
from unittest.mock import patch
from traffic_lights.inference.predict import predict
from test.utils import create_mock_image


class MockModel:
    def eval(self):
        return

    def train(self):
        return

    def to(self, device):
        return


class TestPredict(unittest.TestCase):
    def setUp(self):
        plot_patch = patch(
            "traffic_lights.inference.predict.plot_detection", MagicMock()
        )
        load_patcher = patch(
            "traffic_lights.inference.predict.torch.load",
            MagicMock(return_value=MockModel()),
        )
        device_patcher = patch(
            "traffic_lights.inference.predict.torch.cuda.is_available",
            MagicMock(return_value=False),
        )
        image_patcher = patch(
            "traffic_lights.data.dataset.Image.open",
            MagicMock(return_value=create_mock_image()),
        )
        self.MockPlotDetection = plot_patch.start()
        self.MockTorchLoad = load_patcher.start()
        self.MockCudaAvailable = device_patcher.start()
        self.MockImageOpen = image_patcher.start()

    def test_device_uses_cpu_if_gpu_not_available(self):
        predict("mock_image_path", "mock_model_path")
        self.assertEqual(self.MockPlotDetection.call_args.args[2].type, "cpu")

    def test_device_uses_cuda_if_gpu_available(self):
        self.MockCudaAvailable.return_value = True
        predict("mock_image_path", "mock_model_path")
        self.assertEqual(self.MockPlotDetection.call_args.args[2].type, "cuda")

    @patch("traffic_lights.inference.predict.torch.device")
    def test_torch_load_called_with_correct_params(self, mock):
        mock.return_value = "cpu"
        predict("mock_image_path", "mock_model_path")
        self.MockTorchLoad.assert_called_once_with(
            "mock_model_path", map_location="cpu"
        )


if __name__ == "__main__":
    unittest.main()
