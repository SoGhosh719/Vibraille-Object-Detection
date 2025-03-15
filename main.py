import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchvision.transforms as transforms

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load MiDaS depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame)

    # Convert frame for depth estimation
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        depth_map = midas(img)
        depth_map = depth_map.squeeze().cpu().numpy()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2, conf, cls = box.xyxy[0].cpu().numpy()
            label = result.names[int(cls)]

            # Calculate estimated distance from depth map
            depth_value = np.mean(depth_map[int(y1):int(y2), int(x1):int(x2)])

            # Convert depth value to estimated distance
            distance_cm = 100 * (1 / (depth_value + 0.001))  # Adjust scaling factor

            # Only process objects within 1m range
            if distance_cm <= 100:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {int(distance_cm)} cm", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Object Detection with Depth Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
import subprocess  # For sound-based feedback

def play_tone(duration):
    """
    Simulates a vibration effect by playing a short silent audio using FFmpeg.
    """
    subprocess.run(["ffplay", "-nodisp", "-autoexit", "-t", str(duration/1000), "-loglevel", "quiet"], check=False)

# In the loop where we detect objects within 1m, add:
if distance_cm <= 100:
    intensity = int((100 - distance_cm) * 2.5)  # Stronger vibration for closer objects
    print(f"Object detected: {label}, Distance: {int(distance_cm)} cm, Vibration: {intensity}")
    play_tone(intensity)  # Simulated vibration feedback
