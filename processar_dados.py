import cv2
import os
import numpy as np

# Definir caminho do dataset
dataset_path = './dataset/fall-detection-dataset/'

def processar_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Processar todos os vídeos do dataset
frames = []
for video_file in os.listdir(dataset_path):
    video_path = os.path.join(dataset_path, video_file)
    if video_file.endswith('.mp4'):
        frames.extend(processar_video(video_path))
