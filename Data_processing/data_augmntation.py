import os
import torch
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import numpy as np

# 设置路径
SOURCE_DIR = './pt_file_save'
TARGET_DIR = './pt_file_augmented'
os.makedirs(TARGET_DIR, exist_ok=True)

# 设置增强器
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5)
])

# 设置切段参数
SEGMENT_LEN = 64000
OVERLAP = 0.5  # 50% overlap

# 遍历 crackle 标签的 pt 文件进行增强和切段
for fname in os.listdir(SOURCE_DIR):
    if not fname.endswith('.pt'):
        continue
    path = os.path.join(SOURCE_DIR, fname)
    data = torch.load(path)

    waveform = data['waveform']  # [1, T]
    label = data['label']        # tensor([1, 0])  # crackle = 1

    if label[0] != 1:  # 仅增强 crackle
        continue


    waveform_np = waveform.squeeze().cpu().numpy().astype(np.float32)  
    augmented = augment(samples=waveform_np, sample_rate=16000)       

    augmented_tensor = torch.tensor(augmented).unsqueeze(0)

    # 切段
    T = augmented_tensor.shape[-1]
    stride = int(SEGMENT_LEN * (1 - OVERLAP))
    num_segments = (T - SEGMENT_LEN) // stride + 1

    for i in range(num_segments):
        start = i * stride
        end = start + SEGMENT_LEN
        segment = augmented_tensor[:, start:end]
        if segment.shape[-1] < SEGMENT_LEN:
            continue

        new_fname = f"{fname[:-3]}_aug{i}.pt"
        torch.save({'waveform': segment, 'label': label}, os.path.join(TARGET_DIR, new_fname))
