import sys
import pathlib
import torch
import torchvision.transforms as T
from torchvision.io import read_image
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.sample import config
from components.models import ImgRecogModel

###
# sample
###

filepath_img = "/workspace/data/img/sample_0.png"
img = read_image(filepath_img)

transforms = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ConvertImageDtype(torch.float),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_preprocessed = transforms(img).unsqueeze(dim=0)

# model
model = ImgRecogModel(config["model"])
result = model(img_preprocessed)
