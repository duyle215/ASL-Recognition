from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import base64
import os

#from packages.CustomVGG19 import CustomVGG19
from packages.CustomDeepLabV3 import CustomDeepLabV3
import packages.Resnet50_FineTuning as ResNet50

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

segment_model = CustomDeepLabV3(num_classes=2, pretrained_backbone=True)
segment_model.load_state_dict(torch.load("models/deeplab_segmet_model.pth", map_location=device), strict=False)
segment_model.to(device).eval()

#class_model = CustomVGG19(num_classes=5)
#state_dict = torch.load("models/vgg19_hand_classification.pth", map_location=device)

class CustomFCHead(nn.Module):
    def __init__(self, in_features=2048, num_classes=5, dropout=0.5):
        super(CustomFCHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ResNet50 đã bao gồm AdaptiveAvgPool2d, nên x đã là (batch_size, in_features)
        return self.classifier(x)

# --- Load ResNet50 classification model ---
class_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # Sử dụng weights API mới
num_ftrs = class_model.fc.in_features # Get input features of the original fc layer (should be 2048)
class_model.fc = CustomFCHead(in_features=num_ftrs, num_classes=5, dropout=0.5) # Thay thế bằng custom head
state_dict = torch.load("models/model_resnet50_fine_tuning.pth", map_location=device)
class_model.load_state_dict(state_dict)
class_model.to(device).eval()

def pad_to_square(img):
    w, h = img.size
    max_side = max(w, h)
    new_img = Image.new("RGB", (max_side, max_side), (0, 0, 0))
    new_img.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    return new_img

def crop_center_square(img_np):
    h, w, _ = img_np.shape
    min_side = min(h, w)
    top = (h - min_side) // 2
    left = (w - min_side) // 2
    return img_np[top:top+min_side, left:left+min_side]

seg_transform = transforms.Compose([
    transforms.Lambda(crop_center_square),
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

class_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

CLASSES = ['U', 'V', 'W', 'X', 'Y']

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream).convert("RGB")
    img_np = np.array(img)

    input_tensor = seg_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_output = segment_model(input_tensor)['out']
    mask = torch.argmax(seg_output.squeeze(), dim=0).byte().cpu().numpy()

    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    padding = 1.0
    x_pad = int(w * padding)
    y_pad = int(h * padding)
    x1 = max(x - x_pad, 0)
    y1 = max(y - y_pad, 0)
    x2 = min(x + w + x_pad, img_np.shape[1])
    y2 = min(y + h + y_pad, img_np.shape[0])
    hand_roi_rgb = img_np[y1:y2, x1:x2]
    hand_pil = Image.fromarray(hand_roi_rgb).resize((200, 200))

    input_class = class_transform(hand_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = class_model(input_class)
        probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()
    result = CLASSES[pred_class]

    hand_img_bgr = cv2.cvtColor(np.array(hand_pil), cv2.COLOR_RGB2BGR)
    mask_rgb = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    return jsonify({
        'original_img': image_to_base64(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)),
        'hand_img': image_to_base64(hand_img_bgr),
        'prediction': result,
        'confidence': round(confidence * 100, 2),
        'mask_img': image_to_base64(mask_rgb)
    })

if __name__ == '__main__':
    app.run(debug=True)
