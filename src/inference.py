import torch

from PIL import Image

from model import CatsDogsModel
from config import *

import torchvision.transforms.v2 as tfs

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transforms = tfs.Compose([
    tfs.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    tfs.ToImage(),
    tfs.ToDtype(torch.float32, scale=True)
])

model = CatsDogsModel().to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model.eval()

image = Image.open("cat.jpg").convert("RGB")

image = transforms(image)

image = image.unsqueeze(0).to(device)

with torch.no_grad():

    outputs = model(image)

    pred = torch.argmax(outputs, dim=1)

label = "Cat" if pred.item() == 0 else "Dog"

print(f"Prediction: {label}")
