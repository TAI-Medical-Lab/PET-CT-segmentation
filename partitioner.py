import torch
import torch.distributed as dist


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

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        part_0 = []
        part_1 = []
        part_2 = []
        part_3 = []
        for index in indexes:
            if index % 4 == 0:
                part_0.append(index)
            elif index % 4 == 1:
                part_1.append(index)
            elif index % 4 == 2:
                part_2.append(index)
            elif index % 4 == 3:
                part_3.append(index)

        self.partitions.append(part_0)
        self.partitions.append(part_1)
        self.partitions.append(part_2)
        self.partitions.append(part_3)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def partition_dataset(dataset, batch_size):
    size = dist.get_world_size()
    bsz = int(batch_size / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    loader = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True,num_workers=4,pin_memory=True)
    return loader, bsz