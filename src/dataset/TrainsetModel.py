import os, glob
import torch

class TrainsetModel(torch.utils.data.Dataset):
    def __init__(self, root,target,form, hp):
        self.data_list = [x for x in glob.glob(os.path.join(root,target,form),recursive=True) if not os.path.isdir(x)]

    def __getitem__(self, index):
        data_item = self.data_list[index]

        # Process data_item if necessary.

        data = data_item

        return data

    def __len__(self):
        return len(self.data_list)


