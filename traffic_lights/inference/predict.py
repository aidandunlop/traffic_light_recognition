import torch
import torchvision
from PIL import Image
from .utils import plot_detection


def predict(image_path, model_path):
    print(model_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load(model_path, map_location=device)
    print("ooo", model.eval)
    model.eval()
    model.to(device)
    to_tensor = torchvision.transforms.ToTensor()
    image = Image.open(image_path).convert("RGB")
    plot_detection(to_tensor(image), model, device, 0.8, rect_th=2)
