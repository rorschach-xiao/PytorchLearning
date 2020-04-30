import torch
import numpy as np
from torch.utils.data import DataLoader,IterableDataset

# example 1 在__iter__方法中实现对多进程loading任务的分发
class MyIterDataset(IterableDataset):
    def __init__(self,start,end,**kwargs):
        super(MyIterDataset,self).__init__(**kwargs)
        assert end>start
        self.start = start
        self.end = end

    def __iter__(self):
        work_info = torch.utils.data.get_worker_info()
        if work_info is None:  # 如果是单进程，则直接返回整个数据集
            iter_start = self.start
            iter_end = self.end
        else:  # 如果是多进程，则将数据集平均分配个每个进程
            per_worker = int(np.ceil((self.end-self.start)/float(work_info.num_workers)))
            worker_id = work_info.id
            iter_start = self.start + worker_id*per_worker
            iter_end = min(iter_start+per_worker,self.end)
        return iter(range(iter_start,iter_end))

dataset = MyIterDataset(start=4,end=8)
dataloader_1 = DataLoader(dataset,num_workers=0) # 单进程
dataloader_2 = DataLoader(dataset,num_workers=2) # 双进程 进程1负责加载（4，5） 进程2负责加载（6，7）
print(list(dataloader_1))
#[tensor([4]), tensor([5]), tensor([6]), tensor([7])]
print(list(dataloader_2))
#[tensor([4]), tensor([6]), tensor([5]), tensor([7])]

# example 2 通过worker_init_fn方法实现多进程loading任务的分发

class MyIterDataset2(IterableDataset):
    def __init__(self,start,end):
        super(MyIterDataset2,self).__init__()
        assert end>start
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(range(self.start,self.end))
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end # configure the dataset to only process the split workload
    per_worker = int(np.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)

dataset = MyIterDataset2(start =4,end = 10)
dataloader_3 = DataLoader(dataset,num_workers=0)
dataloader_4 = DataLoader(dataset,num_workers=2)
dataloader_5 = DataLoader(dataset,num_workers=2,worker_init_fn=worker_init_fn)
print(list(dataloader_3))
# [tensor([4]), tensor([5]), tensor([6]), tensor([7]), tensor([8]), tensor([9])]
print(list(dataloader_4))
# [tensor([4]), tensor([4]), tensor([5]), tensor([5]), tensor([6]), tensor([6]), tensor([7]), tensor([7]), tensor([8]), tensor([8]), tensor([9]), tensor([9])]
print(list(dataloader_5))
# [tensor([4]), tensor([7]), tensor([5]), tensor([8]), tensor([6]), tensor([9])]





















