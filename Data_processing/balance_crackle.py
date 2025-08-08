import os
import torch
import shutil
import random


#This script balances the number of crackle-labeled samples with wheeze-labeled samples


DATA_DIR = './pt_segments'


wheeze_files = []
crackle_files = []

for fname in os.listdir(DATA_DIR):
    if not fname.endswith('.pt'):
        continue
    fpath = os.path.join(DATA_DIR, fname)
    try:
        data = torch.load(fpath)
        if data['label'][0] == 1:
            wheeze_files.append(fname)
        if data['label'][1] == 1:
            crackle_files.append(fname)
    except Exception as e:
        print(f"Skip {fname}: {e}")


n_wheeze = len(wheeze_files)
n_crackle = len(crackle_files)
print(f" wheeze ：{n_wheeze}")
print(f" crackle ：{n_crackle}")

# 增强 crackle 使数量与 wheeze 相等
if n_crackle >= n_wheeze:
    print("no action needed")
else:
    copies_needed = n_wheeze - n_crackle
    print(f"copying {copies_needed} crackles")

    for i in range(copies_needed):
        src_file = random.choice(crackle_files)
        base = os.path.splitext(src_file)[0]
        new_name = f"{base}_aug{i}.pt"
        src_path = os.path.join(DATA_DIR, src_file)
        dst_path = os.path.join(DATA_DIR, new_name)
        shutil.copyfile(src_path, dst_path)

    print(f"complete, the crackle amount is  {n_wheeze}")
