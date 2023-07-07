import numpy as np
from torch.utils.data import Sampler


class BalancedBatchSampler(Sampler):
    def __init__(self,
                 cfg,
                 dataset):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.cfg = cfg
        self.dataset = dataset

        self.normal_generator = self.randomGenerator(self.dataset.normal_idx)
        self.anomaly_generator = self.randomGenerator(self.dataset.anomaly_idx)
        # n_normal: 1/2; n_anomaly: 1/2
        if self.cfg.num_anomalies != 0:
            self.n_normal = self.cfg.batch_size // 2
            self.n_anomaly = self.cfg.batch_size - self.n_normal
        else:
            self.n_normal = self.cfg.batch_size
            self.n_anomaly = 0

    def randomGenerator(self, list):
        while True:
            random_list = np.random.permutation(list)
            for i in random_list:
                yield i
    
    def __len__(self):
        return self.cfg.steps_per_epoch
    
    def __iter__(self):
        for _ in range(self.cfg.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_anomaly):
                batch.append(next(self.outlier_generator))

            yield batch


#train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=BalancedBatchSampler(args, train_dataset), **kwargs)