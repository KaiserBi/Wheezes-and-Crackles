import os 
import torch 
from torch.utils.data import Dataset

class myDataSet(Dataset):

    def __init__ (self, root_dir):
      self.file = []
      for f in os.listdir(root_dir):
         if f.endswith(".pt"):
            self.file.append(os.path.join(root_dir, f))
    
    def __len__ (self):
       return len(self.file)
    
    def __getitem__(self, idx):
       try:
        item = torch.load(self.file[idx])
        x = item['waveform'].unsqueeze(0)  # → shape: [1, T] for Conv1D
        x = pad_to_same(x)
        y = item['label'].float()
        return x, y
       except Exception as e:
        print(f"⚠️ 跳过损坏样本 {self.file[idx]}: {e}")
        return None

    

def pad_to_same(x, target_len=80000):
   if(x.shape[-1]<target_len):
      pad_len = target_len - x.shape[-1]
      return torch.nn.functional.pad(x, (0,pad_len))
   elif(x.shape[-1]>target_len):
      x=x[:,:target_len]
      return x
   else:
      return x
   






