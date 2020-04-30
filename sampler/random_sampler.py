from torch.utils.data import sampler
from torch.utils.data import DataLoader
from dataset.MapDataset import MyMapDataset
import numpy as np


if __name__=="__main__":
    '''
    当使用RandomSampler时，有两个要注意的参数：
    1、 replacement (bool) 该参数为True时可以进行重复采样，否则不行。default=False
    2、 num_samples (int) 该参数表示从数据集中采样的样本数量，当设置为特定值时，replacement应设为True. default=len(dataset)
    '''
    imgs=[]
    labels=[]
    for i in range(3):
        imgs.append(np.random.randint(5,size=(3,3,3)))
        labels.append(i)
    dataset = MyMapDataset(imgs,labels)
    dataloader_1 = DataLoader(dataset,sampler=sampler.RandomSampler(dataset))
    dataloader_2 = DataLoader(dataset,sampler=sampler.RandomSampler(dataset,replacement=True))
    dataloader_3 = DataLoader(dataset,sampler=sampler.RandomSampler(dataset,replacement=True,num_samples=2))
    print("=======This is dataloader_1=======")
    for index,item in enumerate(dataloader_1):
        image,label = item
        print("This is No.%d sample: " % index)
        print(label)
    print("=======This is dataloader_2=======")
    for index, item in enumerate(dataloader_2):
        image, label = item
        print("This is No.%d sample: " % index)
        print(label)
    print("=======This is dataloader_3=======")
    for index,item in enumerate(dataloader_3):
        image,label = item
        print("This is No.%d sample: " % index)
        print(label)
'''
=======This is dataloader_1=======
This is No.0 sample: 
tensor([1])
This is No.1 sample: 
tensor([2])
This is No.2 sample: 
tensor([0])
=======This is dataloader_2=======
This is No.0 sample: 
tensor([2])
This is No.1 sample: 
tensor([1])
This is No.2 sample: 
tensor([2])
=======This is dataloader_3=======
This is No.0 sample: 
tensor([0])
This is No.1 sample: 
tensor([1])

'''