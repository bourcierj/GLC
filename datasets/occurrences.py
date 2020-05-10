import pandas as pd
import torch
from torch.utils.data import Dataset

from data_utils.read import load_occurrences_data


class OccurrencesDataset(Dataset):
    """
    A dataset for loading occurrences data.
    Args:
        path_csv (str): path to the csv file.
        reveals_targets (bool, default: True): bool indicating if the target species are available as a column.
        filter: optional filter object to apply on occurrences data during __init__.
        transform optional transform object to apply on a data sample during __getitem__.

    Attributes:
        occurrences (pandas.DataFrame)
        targets (pandas.Series, optional)
        orig_targets (pandas.Series, optional)
        orig_target_names (pandas.Series, optional)
        num_classes (int, optional)
        class_index_to_species_id (pandas.Series, optional)
        species_id_to_class_index (pandas.Series, optional)
        transform (list, optional)
    """
    def __init__(self, path_csv, reveal_targets=True, filter=None, transform=None):

        # the occurrences data in a dataframe
        self.occurrences = load_occurrences_data(path_csv, reveal_targets)
        # apply optional filter(s) on occurrences data
        if filter is not None:
            self.occurrences = filter(self.occurrences)

        self.targets = None
        # if target are given a few steps are needed to treat them properly:
        if reveal_targets:
            self.orig_targets = self.occurrences.pop('species_id')
            self.orig_target_names = self.occurrences.pop('GBIF_species_name')
            # critical!
            assert 'species_id' not in self.occurrences.columns
            assert 'GBIF_species_name' not in self.occurrences.columns

            # get classes indices
            species_ids_array = self.orig_targets.unique()
            # get number of classes
            self.num_classes = len(species_ids_array)

            self.class_index_to_species_id = pd.Series(
                species_ids_array, index=pd.RangeIndex(self.num_classes)
            )
            self.species_id_to_class_index = pd.Series(
                self.class_index_to_species_id.index.values,
                index=self.class_index_to_species_id.values
            )
            # encode targets
            self.targets = self.orig_targets.map(self.species_id_to_class_index)
            # important!
            assert (self.targets.index == self.occurrences.index).all()

        self.transform = transform

    def __getitem__(self, index):
        """Returns:
            tensor: the sample data, a tensor of shape (2,) with lat,lon values
            tensor: the sample target (class label index)
        """
        data = self.occurrences.iloc[index].values
        target = self.targets.iloc[index]

        # apply optional transform(s) on data
        if self.transform is not None:
            data = self.transform(data)

        return torch.tensor(data), torch.tensor(target)

    def __len__(self):
        return len(self.occurrences.index)

    def decode_targets(self, targets):
        """Get original species id from class indices. Useful for predicting."""
        if isinstance(targets, torch.Tensor):
            # get as tensor
            return torch.tensor(
                self.class_index_to_species_id[targets.numpy()],
                dtype=targets.dtype
            )
        # get as pandas series
        targets = self.class_index_to_species_id[targets].values
        return targets

if __name__ == '__main__':
    from constants import PATH_FR_TRAIN
    dataset = OccurrencesDataset(PATH_FR_TRAIN)
