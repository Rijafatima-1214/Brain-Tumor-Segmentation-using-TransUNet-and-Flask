import torch
from torchvision import transforms
from PIL import Image
import os

# Load model
from transunet import TransUNet   # <- or wherever your model class is defined

# 1️⃣ Initialize the model architecture
model = TransUNet(img_size=224, num_classes=4)  # adjust img_size/classes as used in training

# 2️⃣ Load the saved weights
state_dict = torch.load('model/transunet_checkpoint.pth', map_location='cpu')
model.load_state_dict(state_dict)

# 3️⃣ Set to evaluation mode
model.eval()


def predict_mask(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    x = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        y_pred = model(x)
    # TODO: Postprocess y_pred into a mask image
    # For now, just return uploaded image as placeholder
    return image_path
