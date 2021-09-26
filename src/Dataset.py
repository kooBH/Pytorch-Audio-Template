import os, glob
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, hp, is_train=True):
        self.hp = hp
        self.root = hp.data.root

        if is_train :
            self.list_data= [x for x in glob.glob(os.path.join(self.root,'train','*.pt'))]
        else :
            self.list_data= [x for x in glob.glob(os.path.join(self.root,'test','*.pt'))]



    def __getitem__(self, index):
        data_item = self.list_data[index]

        # Process data_item if necessary.

        data = data_item

        return data

    def __len__(self):
        return len(self.data_list)


