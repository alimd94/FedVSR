

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from typing import Union
import json
from collections import defaultdict


def load_datasets(  # pylint: disable=too-many-arguments
    tr_s,
    ts_s,
    num_clients: int = 10,
    iid: Optional[bool] = True,
    balance: Optional[bool] = True,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 123,
    alpha: Optional[float] = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates the dataloaders to be fed into the model.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    iid : bool, optional
        Whether the data should be independent and identically distributed between the
        clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default True
    balance : bool, optional
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader for testing.
    """
    datasets, testset = _partition_data(tr_s, ts_s, num_clients, iid, balance, seed,alpha=alpha)
    

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)


def _partition_data(
    trainset,
    testset,
    num_clients: int = 10,
    iid: Optional[bool] = True,
    balance: Optional[bool] = True,
    seed: Optional[int] = 123,
    alpha: Optional[float] = 0,
    name: Optional[str] = None
) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid or non iid partitions to simulate the
    federated setting.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    iid : bool, optional
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default True
    balance : bool, optional
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the model.
    """
    num_clients=int(num_clients)

    if balance:
        if iid:
            partition_size = int(len(trainset) // int(num_clients))
            lengths = [partition_size] * num_clients
            # datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))

             #to simulate the real world scenario, each client has whole videos,, shuffeling here split the videos into parts then assign.

            datasets = [Subset(trainset, range(i * partition_size, (i + 1) * partition_size)) for i in range(int(num_clients))]

        else:
            if alpha > 0:
                x = [i*alpha for i in np.linspace(start=1, stop=num_clients, num=num_clients)]
                data_division_in_units = np.exp(x)
                total_units= data_division_in_units.sum()
                # Normalize the units to get the fraction total dataset
                partition_sizes_as_fraction = data_division_in_units / total_units
                # Calculate the number of samples
                partition_sizes_as_num_of_samples = np.array(partition_sizes_as_fraction * len(trainset) , dtype=np.int64)
                # Check if any sample is not allocated because of multiplication with fractions.
                assigned_samples = np.sum(partition_sizes_as_num_of_samples)
                left_unassigned_samples = len(trainset) - assigned_samples
                # If there is any sample(s) left unassigned, assign it to the largest partition.
                partition_sizes_as_num_of_samples[-1] += left_unassigned_samples
                indices = np.arange(len(trainset))
                start_idx = 0
                datasets = []
                for size in partition_sizes_as_num_of_samples:
                    end_idx = start_idx + size
                    subset_indices = indices[start_idx:end_idx]
                    datasets.append(Subset(trainset, subset_indices))
                    start_idx = end_idx
                
    else:
        if name != 'kinetics':
            raise ValueError("Non-IID partitioning is only supported for the Kinetics dataset, REDS has no labels.")

        with open('updated_file.json', 'r') as f:
            loaded_lookup = json.load(f)
        adjusted_video_labels = {}
        indices_to_remove=[]
        new_index = 0
        for old_index in range(240):
            if old_index not in indices_to_remove:
                adjusted_video_labels[new_index] = loaded_lookup[str(old_index)]
                new_index += 1

        label_to_indices = defaultdict(list)
        for idx, label in adjusted_video_labels.items():
            start_frame = (int(idx)) * 100  
            end_frame = start_frame + 100
            label_to_indices[label].extend(range(start_frame, end_frame))
        datasets = []

        for label, indices in label_to_indices.items():
            for i in range(0, len(indices), 600):
                subset_indices = indices[i:i + 600]
                if len(subset_indices) == 600:
                    datasets.append(Subset(trainset, subset_indices))

        # for i,dataset in enumerate(datasets):
        #     with open("copy.txt", "a") as myfile:
        #         myfile.write(str(len(trainset)))
        #         myfile.write("+++++++++++++++++++++")
        #         for _ in dataset.indices:
        #                 myfile.write("\n")
        #                 myfile.write((str(i)+','+str(_)))

    return datasets, testset


def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 123,
) -> Dataset:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    Dataset
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in class_counts:
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(trainset.targets[idxs[int(count) : int(count + smallest)]])
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled
