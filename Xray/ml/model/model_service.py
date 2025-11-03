import io
import bentoml
import numpy as np
import torch
import torch.nn as nn
from bentoml.io import Image, Text
from PIL import Image as PILImage
from Xray.constant.training_pipeline import *
from Xray.ml.model.arch import Net  # your model

# ✅ Allow all safe PyTorch modules
torch.serialization.add_safe_globals([
    Net,
    nn.Sequential,
    nn.Conv2d,
    nn.Linear,
    nn.ReLU,
    nn.MaxPool2d,
    nn.Flatten,
    nn.Dropout,
    nn.BatchNorm2d,
    nn.AvgPool2d
])

bento_model = bentoml.pytorch.get(BENTOML_MODEL_NAME)
runner = bento_model.to_runner()
svc = bentoml.Service(name=BENTOML_SERVICE_NAME, runners=[runner])

@svc.api(input=Image(allowed_mime_types=["image/jpeg"]), output=Text())
async def predict(img):
    b = io.BytesIO()
    img.save(b, "jpeg")
    im_bytes = b.getvalue()
    my_transforms = bento_model.custom_objects.get(TRAIN_TRANSFORMS_KEY)
    image = PILImage.open(io.BytesIO(im_bytes)).convert("RGB")
    image = torch.from_numpy(np.array(my_transforms(image).unsqueeze(0)))
    image = image.reshape(1, 3, 224, 224)
    batch_ret = await runner.async_run(image)
    pred = PREDICTION_LABEL[max(torch.argmax(batch_ret, dim=1).detach().cpu().tolist())]
    return pred

