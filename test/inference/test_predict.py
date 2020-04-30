import unittest
from unittest.mock import MagicMock, patch
from traffic_lights.inference.predict import predict, load_model
from test.utils import create_mock_image


class MockModel:
    def eval(self):
        return

    def train(self):
        return

    def to(self, device):
        return


class TestLoadModel(unittest.TestCase):
    def setUp(self):
        load_patcher = patch(
            "traffic_lights.inference.predict.torch.load",
            MagicMock(return_value=MockModel()),
        )
        cuda_available_patcher = patch(
            "traffic_lights.inference.predict.torch.cuda.is_available",
            MagicMock(return_value=False),
        )
        torch_device_patcher = patch(
            "traffic_lights.inference.predict.torch.device",
            MagicMock(return_value="cpu"),
        )
        self.MockTorchLoad = load_patcher.start()
        self.MockCudaAvailable = cuda_available_patcher.start()
        self.TorchDevice = torch_device_patcher.start()

    def test_device_uses_cpu_if_gpu_not_available(self):
        load_model("mock_model_path")
        self.assertEqual(load_model("mock_model_path"), (self.MockTorchLoad(), "cpu"))

    def test_device_uses_cuda_if_gpu_available(self):
        self.TorchDevice.return_value = "cuda"
        self.MockCudaAvailable.return_value = True
        self.assertEqual(load_model("mock_model_path"), (self.MockTorchLoad(), "cuda"))

    def test_torch_load_called_with_correct_params(self):
        load_model("mock_model_path")
        self.MockTorchLoad.assert_called_once_with(
            "mock_model_path", map_location="cpu"
        )


if __name__ == "__main__":
    unittest.main()
