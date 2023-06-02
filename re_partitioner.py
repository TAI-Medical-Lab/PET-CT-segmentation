import torch
import torch.distributed as dist
import random
import copy


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes=[0.4, 0.25, 0.25, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        part_sizes = [int(data_len * size) for size in sizes]
        print(part_sizes)

        max_part_size = max(part_sizes)

        start = 0
        for i, part_size in enumerate(part_sizes):
            if part_size < max_part_size:
                end = start + part_size
                additional_count = max_part_size - part_size
                additional_indexes = (indexes[start:end] * (additional_count // part_size + 1))[:additional_count]
                self.partitions.append(indexes[start:end] + additional_indexes)
            else:
                end = start + part_size
                self.partitions.append(indexes[start:end])
            
            start = end

        #print(self.partitions)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])




def partition_dataset(dataset0, dataset1, dataset2, dataset3, batch_size):
    size = dist.get_world_size()
    bsz = int(batch_size / float(size))

    len0=len(dataset0)
    len1=len(dataset1)
    len2=len(dataset2)
    len3=len(dataset3)
    lenall=len0+len1+len2+len3
    partition_sizes = [len0/lenall, len1/lenall, len2/lenall, len3/lenall]

    all_data = torch.utils.data.ConcatDataset([dataset0, dataset1, dataset2, dataset3])
    partition = DataPartitioner(all_data, partition_sizes)
    partition = partition.use(dist.get_rank())
    loader = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True, num_workers=4, pin_memory=True)
    return loader, bsz
