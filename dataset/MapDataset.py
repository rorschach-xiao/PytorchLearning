import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset

class MyMapDataset(Dataset):
    def __init__(self,imgs,labels,**kwargs):
        super(MyMapDataset,self).__init__(**kwargs)
        self.data_list = []
        for img,label in zip(imgs,labels):
            self.data_list.append({
                "img":img,
                "label":label
            })
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]["img"],self.data_list[index]["label"]

if __name__ =='__main__':
    imgs=[]
    labels=[]
    for i in range(3):
        imgs.append( np.random.randint(5,size=(3,3,3)))
        labels.append(i)
    dataset = MyMapDataset(imgs,labels)
    dataloader = DataLoader(dataset,num_workers=2)

    for index,item in enumerate(dataloader):
        image,label = item
        print("This is No.%d sample: "%index)
        print(image)
        print(label)






