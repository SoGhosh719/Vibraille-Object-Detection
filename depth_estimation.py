import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

# Load MiDaS model
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
model.eval()

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        depth_map = model(img)
        depth_map = depth_map.squeeze().cpu().numpy()

    depth_map = (depth_map / np.max(depth_map) * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

    cv2.imshow("Depth Estimation", depth_map)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
