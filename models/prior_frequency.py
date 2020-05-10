import numpy as np
import torch

class PriorFrequencyModel():
    """Prior frequency model baseline.
    This baseline computes the frequency of classes in a training set and predicts the
    most frequent labels in order of decreasing frequency.
    """
    def __init__(self):
        return

    def train(self, data=None, target=None):
        assert target is not None, 'target should be provided.'

        # get classes with their counts
        class_indices, class_counts = torch.unique(
            target, sorted=True, return_counts=True)

        # get class probas (frequencies)
        self.output = class_counts.to(torch.float) / len(target)

    def predict(self, data):
        """Returns prior frequencies of training set labels."""
        return self.output.repeat(data.size(0), 1)

if __name__ == '__main__':

    import random

    from constants import PATH_FR_TRAIN
    from datasets.occurrences import OccurrencesDataset
    from data_utils.utils import process_output_logits

    dataset = OccurrencesDataset(PATH_FR_TRAIN)

    # get full training set in tensors
    from tqdm import tqdm
    train_datas, train_targets = map(
        lambda tensors: torch.stack(tensors, dim=0),
        zip(*tuple(dataset[idx] for idx in tqdm(range(100_000))))
    )

    class Loader():
        """Data batches iterator. Generate different random batches at each epoch"""
        def __init__(self, data, target, batch_size):
            self.batches = [
                (data[i:i+batch_size, :], target[i:i+batch_size])
                for i in range(0, data.size(0)//batch_size, batch_size)
            ]

        def __iter__(self):
            random.shuffle(self.batches)
            return iter(self.batches)

        def __len__(self):
            return len(self.batches)
    #Note: the model can't predict on full training set at once: memory error. Need to
    # iterate in small batches (of size 100).

    model = PriorFrequencyModel()
    model.train(train_datas, train_targets)

    loader = Loader(train_datas, train_targets, batch_size=100)
    for i, (data, target) in enumerate(loader):
        output = model.predict(data)
        processed = process_output_logits(output, prediction_length=30)
        if i > 10:
            break
