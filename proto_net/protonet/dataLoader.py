import torch
from torch.utils.data import Dataset, DataLoader
from proto_net.utils.subs_to_list import subs_to_list
from processing.constants import ALL_TRIAL_LIST
import numpy as np

'''
This function was taken from the github repo of prototypical network
and was modified to fit the input data. If you are using it with your
own data, make sure to modify so that it works with your input.
'''
class ProtoNetDataSet(Dataset):
    def __init__(self,data, query_subs, sup_subs, trials, n_shots, test=False,ft="features"):
        """
        data : tensor
            A tensor containing all data points. Shape [c, q, .., ..]
        n_shots : int
            Number of support examples per class in each episode.
        n_queries : int
            Number of query examples per class in each episode.
        n_classes : int
            Number of classes in each episode.
        """
        # Assign query data from the query subjects
        self.data_q = subs_to_list(data,query_subs, trials, ft)

        # Assign support data from support subjects
        self.data_s = subs_to_list(data,sup_subs, ALL_TRIAL_LIST, ft)

        # Assign the number of shots
        self.n_shots = n_shots


        len_s = self.data_s.size(1)

        # Initialize the support set indices, so that they remain constant when using them for testing
        indices = []
        for i in range(n_shots):
            k = i + 0.5
            ind = np.floor(k*len_s/n_shots)
            indices.append(int(ind))
        self.sup_indices = indices


        # If we are testing, then remove the samples used for the support set
        # from the query set since they were extracted from the same subject
        if test:
            indices_to_keep = [i for i in range(self.data_q.size(1)) if i not in self.sup_indices]
            self.data_q = self.data_q[:,indices_to_keep,:]

        self.n_queries = self.data_q.size(1)
        self.n_class = self.data_q.size(0)
        assert self.n_class == self.data_s.size(0)


    def get_sup(self):
        s_data = self.data_s[:,self.sup_indices,:]
        return s_data

    def __len__(self):
        return self.n_queries

    def __getitem__(self, idx):
        """
        Get one element from each class
        """
        return self.data_q[torch.arange(self.n_class), idx]


class ProtoNetDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.data_s = dataset.get_sup()


    def __iter__(self):
        for batch in super().__iter__():
            sample = {}
            sample['xs'] = self.data_s
            sample['xq'] = batch.movedim(0,1)
            yield sample

