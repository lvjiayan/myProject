import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend

# 新版本加载模型
model = YOLO("yolov8s.pt")

# 包含前处理和后处理的加载模型
# model  = AutoBackend(weights="yolov8s.pt")

# 老版本加载模型
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
### model = torch.hub.load('./yolov5', 'custom', path='./weights/pose.pt',source='local')

# 从摄像头读取视频
cap = cv2.VideoCapture(0)

# 接收从主机摄像头抓取并通过网络发送过来的视频信息
# address = "udp://127.0.0.1:5000"
# cap = cv2.VideoCapture(address)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    ## 左右反向
    frame = cv2.flip(frame, 1)

    ## 使用YOLOv5进行检测
    # results = model(frame)[0]
    # names   = results.names
    # boxes   = results.boxes.data.tolist()

    # for obj in boxes:
    #     left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
    #     confidence = obj[4]
    #     label = int(obj[5])
    #     color = random_color(label)
    #     cv2.rectangle(frame, (left, top), (right, bottom), color=color ,thickness=2, lineType=cv2.LINE_AA)
    #     caption = f"{names[label]} {confidence:.2f}"
    #     w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
    #     cv2.rectangle(frame, (left - 3, top - 33), (left + w + 10, top), color, -1)
    #     cv2.putText(frame, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

    cv2.imshow('YOLOv8', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()