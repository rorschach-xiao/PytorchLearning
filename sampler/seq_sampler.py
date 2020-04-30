from torch.utils.data import sampler
from torch.utils.data import DataLoader
from dataset.MapDataset import MyMapDataset
import numpy as np


if __name__=="__main__":
    '''
    当使用SequentialSampler的时候，dataloader在每一个epoch都会按照固定顺序从dataset中采样，
    故使用时 shuffle 应设置为 false
    '''
    imgs=[]
    labels=[]
    for i in range(3):
        imgs.append(np.random.randint(5,size=(3,3,3)))
        labels.append(i)
    dataset = MyMapDataset(imgs,labels)
    dataloader = DataLoader(dataset,sampler=sampler.SequentialSampler(dataset),shuffle=False)

    for index,item in enumerate(dataloader):
        image,label = item
        print("This is No.%d sample: " % index)
        print(image)
        print(label)

'''
This is No.0 sample: 
tensor([[[[2, 0, 4],
          [3, 2, 4],
          [4, 0, 0]],

         [[1, 4, 3],
          [2, 3, 3],
          [0, 3, 3]],

         [[1, 0, 0],
          [0, 0, 3],
          [2, 2, 2]]]])
tensor([0])
This is No.1 sample: 
tensor([[[[3, 1, 0],
          [1, 4, 2],
          [3, 4, 1]],

         [[0, 2, 1],
          [4, 4, 0],
          [0, 2, 4]],

         [[2, 1, 3],
          [1, 3, 2],
          [3, 2, 3]]]])
tensor([1])
This is No.2 sample: 
tensor([[[[4, 3, 2],
          [0, 4, 0],
          [4, 3, 3]],

         [[2, 3, 3],
          [4, 1, 1],
          [3, 0, 1]],

         [[0, 3, 4],
          [3, 0, 1],
          [3, 4, 1]]]])
tensor([2])
'''
