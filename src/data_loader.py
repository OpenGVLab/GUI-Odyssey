from torch.utils.data import Dataset
import torch
import random
import json, os


class GUIOdysseyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        
        self.dataset = args.dataset
        
        self.data = self.load_GUIOdyssey()
        self.len = len(self.data)
        
        random.shuffle(self.data)
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]
        
    def load_GUIOdyssey(self):
        d = json.load(open(self.dataset))
        return d