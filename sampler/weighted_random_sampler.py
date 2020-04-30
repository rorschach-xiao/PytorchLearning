from torch.utils.data import sampler
from torch.utils.data import DataLoader
from dataset.MapDataset import MyMapDataset
import numpy as np

if __name__=="__main__":
    '''
    WeightedRandomSampler 是按照给定的权重概率，对 list [0,..,len(weights)-1] 中的数进行采样
    当使用WeightedRandomSampler时，有三个要注意的参数：
    1、 weights(sequence) 该参数为权重的序列
    2、 replacement (bool) 该参数为True时可以进行重复采样，否则不行。default=False 
    3、 num_samples (int) 该参数表示从数据集中采样的样本数量，当设置为特定值时，replacement应设为True. default=len(dataset)
    '''

    for i in range(3):
        print(list(sampler.WeightedRandomSampler([0.5, 0.3, 0.1,0.9,0.8], num_samples=4, replacement=False)))
    '''
    [4, 1, 0, 3]
    [3, 1, 4, 0]
    [3, 4, 1, 0]
    '''
    for i in range(3):
        print(list(sampler.WeightedRandomSampler([0.1,3, 0.7,0.4,0.3], num_samples=3, replacement=True)))
    '''
    [2, 1, 2]
    [0, 1, 1]
    [1, 3, 1]
    '''


    '''
    example 对于样本不均衡的数据集的加权采样
    '''
    def calweight(labels):
        classes = np.unique(labels)
        weights = np.zeros(len(labels))
        for c in classes:
            freq = np.sum(np.array(labels)==c)
            weights[np.array(labels)==c] = len(labels)/freq
        return weights

    imgs = []
    labels = [1,0,0,0,0,0,0,0,0,1]
    for i in range(10):
        imgs.append(np.random.randint(5, size=(3, 3, 3)))
    weights = calweight(labels)
    print(weights)
    dataset = MyMapDataset(imgs, labels)
    sampler_ = sampler.WeightedRandomSampler(weights,num_samples=len(weights),replacement=True)
    dataloader_1 = DataLoader(dataset,sampler=sampler_,num_workers=0)
    dataloader_2 = DataLoader(dataset,num_workers=0)

    print("THIS IS SeqSampler")
    for index, item in enumerate(dataloader_2):
        img, label = item
        print("label %d has been sampled" % label)
    '''
    label 1 has been sampled
    label 0 has been sampled
    label 0 has been sampled
    label 0 has been sampled
    label 0 has been sampled
    label 0 has been sampled
    label 0 has been sampled
    label 0 has been sampled
    label 0 has been sampled
    label 1 has been sampled
    若使用普通的采样器，采样出来的样本类别就会出现不均衡现象
    '''

    print("THIS IS WeightedRandomSampler")
    for index,item in enumerate(dataloader_1):
        img,label = item
        print("label %d has been sampled"%label)
    '''
    label 0 has been sampled
    label 1 has been sampled
    label 1 has been sampled
    label 1 has been sampled
    label 0 has been sampled
    label 1 has been sampled
    label 1 has been sampled
    label 1 has been sampled
    label 0 has been sampled
    label 0 has been sampled
    可以看到，使用加权随机采样后，采样出来的类别达到平衡。
    '''




