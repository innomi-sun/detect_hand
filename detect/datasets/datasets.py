import os
from pathlib import Path
import numpy as np

from torch.utils.data import Dataset

class HandClipDataset(Dataset):

    def __init__(self, datasets_dir, num_class, input_width=20, input_height=20, transform=None):

        self.data, self.targets = None, None
        self.num_class = num_class
        self.width, self.height = input_width, input_height

        datasets_anno = os.path.join(datasets_dir, 'annotations')
        for anno_file in os.listdir(datasets_anno):

            if not anno_file.lower().startswith("clip_anno_"): 
                continue

            text = open(os.path.join(datasets_anno, anno_file), 'r').readline()
            data = np.array(text.split(','), dtype=np.uint8)
            data = data.reshape(-1, self.width, self.height, 3)

            if type(self.data).__module__ != np.__name__ and self.data == None:
                self.data = data
            else:
                self.data = np.append(self.data, data, axis=0)

            target = anno_file.split('_')[2]
            assert target.isdigit(), 'Incorrect label in file name.'
            target = int(target)

            # make a target list where target index is 1
            # target is 5 when 10 classes, target_list is [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            targets = np.zeros((data.shape[0], self.num_class))
            targets[:, target:target + 1] = 1

            if type(self.targets).__module__ != np.__name__ and self.targets == None:
                self.targets = targets
            else:
                self.targets = np.append(self.targets, targets, axis=0)

        self.transform = transform
        if type(self.data).__module__ != np.__name__ and self.data == None:
            raise ValueError("No data found.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        if self.transform:
            data = self.transform(self.data[idx])
        else:
            data = self.data[idx]
        
        return data, self.targets[idx]

class LandmarkDiffDataset(Dataset):

    def __init__(self, datasets_dir, num_class, input_width=21, input_height=21, transform=None):

        self.data, self.targets = None, None
        self.num_class = num_class
        self.width, self.height = input_width, input_height

        datasets_anno = os.path.join(datasets_dir, 'annotations')
        for anno_file in os.listdir(datasets_anno):

            if not anno_file.lower().endswith('txt'): 
                continue

            text = open(os.path.join(datasets_anno, anno_file), 'r').readline()
            data = np.array(text.split(','), dtype=np.uint8)
            data = data.reshape(-1, self.width, self.height, 3)

            if type(self.data).__module__ != np.__name__ and self.data == None:
                self.data = data
            else:
                self.data = np.append(self.data, data, axis=0)

            target = anno_file.split('_')[0]
            assert target.isdigit(), 'Incorrect label in file name.'
            target = int(target) - 1

            # make a target list where target index is 1
            # target is 5 when 10 classes, target_list is [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            targets = np.zeros((data.shape[0], self.num_class))
            targets[:, target:target + 1] = 1

            if type(self.targets).__module__ != np.__name__ and self.targets == None:
                self.targets = targets
            else:
                self.targets = np.append(self.targets, targets, axis=0)

        self.transform = transform
        if type(self.data).__module__ != np.__name__ and self.data == None:
            raise ValueError("No data found.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        
        if self.transform:
            data = self.transform(self.data[idx])
        else:
            data = self.data[idx]
        
        return data, self.targets[idx]        