
import torch
import numpy as np
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multilabel_accuracy
from scipy.optimize import minimize
from dataset_creation import myDataSet  # 你需要保证这个文件能被正确导入
from resnet1d import ResNet1D           # 或你自己定义的模型
import json
from tqdm import tqdm

# ========== 参数 ========== #
MODEL_PATH = "tuned_best_f1.pt"  # 训练好的模型
CONFIG_PATH = "fine_tuning_config.json"
PT_DATA_DIR = "path/to/your/pt_dataset"  # 你要替换成实际的 pt 数据集路径
BATCH_SIZE = 64

# ========== 加载配置 ========== #
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 加载模型 ========== #
model = ResNet1D(**config["MODEL_CONFIG"]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ========== 验证集加载 ========== #
dataset = myDataSet(PT_DATA_DIR)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========== 收集 logits 和 labels ========== #
logits_list = []
labels_list = []

with torch.no_grad():
    for x, y in tqdm(val_loader):
        x = x.to(device)
        logits = model(x)
        logits_list.append(logits.cpu())
        labels_list.append(y)

logits = torch.cat(logits_list)
labels = torch.cat(labels_list)

# ========== 阈值优化 ========== #
probs = torch.sigmoid(logits).numpy()
labels_np = labels.numpy()
num_labels = probs.shape[1]

def objective(thresholds):
    preds = np.zeros_like(probs)
    for i in range(num_labels):
        preds[:, i] = probs[:, i] > thresholds[i]
    preds_tensor = torch.tensor(preds)
    labels_tensor = torch.tensor(labels_np)
    acc = multilabel_accuracy(preds_tensor, labels_tensor, num_labels=num_labels).item()
    return -acc  # maximize accuracy => minimize negative accuracy

res = minimize(
    objective,
    x0=np.full((num_labels,), 0.5),
    bounds=[(0.0, 1.0)] * num_labels,
    method='L-BFGS-B'
)

# ========== 输出并保存结果 ========== #
best_thresh = res.x
best_acc = -res.fun

print(f"Best thresholds: {best_thresh}")
print(f"Best multi-label accuracy: {best_acc:.4f}")

with open("best_thresholds.json", "w") as f:
    json.dump({f"label_{i}": float(th) for i, th in enumerate(best_thresh)}, f)
print("Saved to best_thresholds.json")
