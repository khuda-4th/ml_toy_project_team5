from ultralytics import YOLO
import os
import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


PATH = 'datasets'

model = YOLO('yolov8s.pt')
model.train(data=PATH + '/data.yaml', epochs=500,
            patience=20, batch=32, imgsz=608, device=device)

print(len(model.names), model.names)
