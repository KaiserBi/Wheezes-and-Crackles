
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset_creation import myDataSet
from resnet1d import ResNet1D
from focal_loss import FocalLoss
from sklearn.metrics import f1_score
from torcheval.metrics.functional import multilabel_accuracy
import json
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from matplotlib import pyplot as plt 


# ========== 配置 ========== #
CONFIG_PATH = "fine_tuning_config.json"
PRETRAINED_MODEL_PATH = "tuned_best_acc.pt"
NEW_MODEL_SAVE_PATH = "6block_best_acc.pt"
BLOCKS_OLD = 4
BLOCKS_NEW = 6
FREEZE_BLOCKS = 2

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 加载数据 ========== #
dataset = myDataSet(config["PT_DATA_DIR"])
n = len(dataset)
n_train = int(0.8 * n)
n_val = n - n_train
train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train_set, batch_size=config["BATCH_SIZE"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["BATCH_SIZE"], shuffle=False)

# ========== 初始化模型 ========== #
model_old = ResNet1D(n_block=BLOCKS_OLD, **config["MODEL_CONFIG"]).to(device)
model_old.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))

model_new = ResNet1D(n_block=BLOCKS_NEW, **config["MODEL_CONFIG"]).to(device)

# ========== 拷贝前 N 层 block 权重 ========== #
for i in range(FREEZE_BLOCKS):
    model_new.basicblock_list[i].load_state_dict(model_old.basicblock_list[i].state_dict())
    for param in model_new.basicblock_list[i].parameters():
        param.requires_grad = False

# ========== 损失函数与优化器 ========== #
loss_fn = FocalLoss(alpha=1.5, gamma=2.0)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_new.parameters()), lr=1e-5)
scaler = GradScaler()

# ========== 验证函数 ========== #
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

            loss = loss_fn(logits, y)
            total_loss += loss.item()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = multilabel_accuracy(all_preds, all_labels).item()
    f1_crackle = f1_score(all_labels[:, 0], all_preds[:, 0])
    f1_wheeze = f1_score(all_labels[:, 1], all_preds[:, 1])
    return total_loss / len(loader), acc, f1_crackle, f1_wheeze

# ========== 训练函数 ========== #
def train(model, loader):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast():
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

# ========== 主训练流程 ========== #
EPOCHS = config["EPOCHS"]
best_acc = 0

crackle_total=[]
wheeze_total = []
acc_total = []
train_total = []
val_total = [] 

for epoch in range(EPOCHS):
    train_loss = train(model_new, train_loader)

    train_total.append(train_loss)

    val_loss, acc, f1_crackle, f1_wheeze = evaluate(model_new, val_loader)

    val_total.append(val_loss)
    acc_total.append(acc)
    crackle_total.append(f1_crackle)
    wheeze_total.append(f1_wheeze)

    
    
    

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"F1 Crackle: {f1_crackle:.4f} | F1 Wheeze: {f1_wheeze:.4f} | Multi-label Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model_new.state_dict(), NEW_MODEL_SAVE_PATH)
        print(f"✅ New best accuracy model saved at epoch {epoch+1}")


fig, aix = plt.subplots(2,2, figsize=(16,9))
aix[0,0].plot(EPOCHS, train_total, label="Train loss")
aix[0,0].plot(EPOCHS, val_total, label="Validation loss")
aix[0,0].legend()
aix[0,0].set_title("losses")

aix[0,1].plot(EPOCHS,crackle_total,label="crackle")
aix[0,1].plot(EPOCHS,wheeze_total,label="wheeze")
aix[0,1].legend()
aix[0,1].set_title("F1 score")

aix[1,0].plot(EPOCHS,acc_total,label="acc")
aix[1,0].set_title("Multi-label accuracy")
plt.savefig('Transfer_result.jpeg')