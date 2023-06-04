import sys
import pathlib
import numpy as np
import torch
import traceback

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.sample import config
from components.models import EfficientNetModel

###
# sample
###
# prepare input
img = np.random.randn(224, 224, 3)
img_tensor = torch.tensor(img)              # to_tensor
img_tensor = img_tensor.permute([2, 0, 1])  # [H,W,C] to [C,H,W]
batch = img_tensor.unsqueeze(dim=0)         # [C,H,W] to [N,C,H,W]
batch = batch.to(torch.float32)             # to_float32

# inference
try:
    model = EfficientNetModel(config["model"])
    out = model(batch.squeeze(dim=1))
    print(out)
except:
    print(traceback.format_exc())
