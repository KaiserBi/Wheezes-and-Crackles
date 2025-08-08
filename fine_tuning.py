import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split
from dataset_creation import myDataSet 
from resnet1d import ResNet1D
from tqdm import tqdm
from early_stopping import EarlyStopping
from torcheval.metrics.functional import multilabel_accuracy
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from torch.cuda.amp import autocast, GradScaler
from focal_loss import FocalLoss


# 读取 JSON 配置文件
with open('fine_tuning_config.json', 'r') as f:
    config = json.load(f)

# 设备设置
device = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")

# POS_WEIGHT 转换为 tensor
POS_WEIGHT = torch.tensor(config["POS_WEIGHT"]).to(device)

BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
LEARNING_RATE = 1e-6



def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.default_collate(batch)

wheeze_count = 2370
crackle_count = 1392
total = 6898





dataset = myDataSet(config["PT_DATA_DIR"])
n = len(dataset)
#print(n)


n_train = int(0.8*n)
n_val = n - n_train

train_set, val_set = random_split(dataset,[n_train, n_val,])




train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle= True,num_workers = 4,  collate_fn=custom_collate,drop_last = False)
val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = True,collate_fn=custom_collate, drop_last=False)


#criteron = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
criteron = FocalLoss(alpha=1.5, gamma=2.0, reduction='mean')


model = ResNet1D(**config["MODEL_CONFIG"]).to(device)
model.load_state_dict(torch.load('fine_tune1.pt'))


optimizer = torch.optim.Adam( model.parameters(),lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min')
scaler = GradScaler()

#early_stop = EarlyStopping(**config["early_stop"])


def train():
    train_loss = [] 
    validation_loss = [] 
    f1_wheeze=[]
    f1_crackle=[]
    val_acc = [] 
    epochs=[]
    
    best_acc=0

    for i in range(1, EPOCHS+1):
        model.train()
        total_loss=0 
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast():
                logits = model(x)
                loss = criteron(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        train_loss.append(total_loss / len(train_loader))

        print(f"[Epoch {i}] Train Loss: {total_loss / len(train_loader):.4f}")

        val_loss, f1wheeze, f1crackle, acc= evaluate(val_loader, split="val")
        scheduler.step(val_loss)
        validation_loss.append(val_loss)
        f1_wheeze.append(f1wheeze)
        f1_crackle.append(f1crackle)
        val_acc.append(acc)
        epochs.append(i)

        
        if(acc>best_acc):
            best_acc= acc
            torch.save(model.state_dict(),"tuned_best_acc.pt")
        



        #early_stop(val_loss, model)
        #if(early_stop.early_stop):
           # print("EARLY STOPPING TRIGGERED")
            #break



    fig, aix = plt.subplots(2,2, figsize=(16,9))
    aix[0,0].plot(epochs, train_loss, label="Train loss")
    aix[0,0].plot(epochs, validation_loss, label="Validation loss")
    aix[0,0].legend()
    aix[0,0].set_title("losses")

    aix[0,1].plot(epochs,f1_crackle,label="crackle")
    aix[0,1].plot(epochs,f1_wheeze,label="wheeze")
    aix[0,1].legend()
    aix[0,1].set_title("F1 score")

    aix[1,0].plot(epochs,val_acc,label="acc")
    aix[1,0].set_title("Multi-label accuracy")
    plt.savefig('Tune_result.jpeg')




def evaluate(loader, split='Val'):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            pred = probs.clone()
            pred[:, 0] = (probs[:, 0] > 0.5).float()  # wheeze
            pred[:, 1] = (probs[:, 1] > 0.5).float()  # crackle

            all_preds.append(pred.cpu())
            all_labels.append(y.cpu())

            loss = criteron(logits, y)
            total_loss += loss.item()
    
    avg_loss = (total_loss/len(loader))

    print(f"Validation loss: {avg_loss}")

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    
    f1_wheeze = f1_score(all_labels[:,0],all_preds[:,0])
    f1_crackle = f1_score(all_labels[:,1],all_preds[:,1])
    print(f"F1_wheeze:{f1_wheeze}")
    print(f"F1_crackle:{f1_crackle}")
    acc = multilabel_accuracy(all_preds,all_labels)
    print(f"Multi-label accuracy:{acc}")

    return avg_loss, f1_wheeze, f1_crackle, acc








if __name__ == '__main__':
   train()


