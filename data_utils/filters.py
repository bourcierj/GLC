import os
import sys

import numpy as np
import pandas as pd


class Compose():
    """A sequence of filters to apply iteratively on a dataframe of occurrences.
    Args:
        filters (list): list of filters instances to apply in order"""
    def __init__(self, filters):
        self.filters = filters
    def __call__(self, df):
        for filter_ in self.filters:
            df = self.filter(df)
        return df


def duplicates(df, filter_out=True):
    """Operation to filter duplicates, i.e samples with the same geoloc and same specie id."""
    mask = df.duplicated(keep=False)
    if filter_out:
        mask = ~mask
    return df[mask].copy()


class Duplicates():
    def __init__(self, filter_out=True):
        self.filter_out = filter_out

    def __call__(self, df):
        return duplicates(df, self.filter_out)


def rare_species(df, frequency_threshold=10, target_column='species_id', filter_out=True):
    """Operation to filter rare species, i.e species with a frequency that is below a threshold."""
    target_serie = df[target_column]
    # get frequency of specie for every occurrence
    counts = target_serie.map(target_serie.value_counts())
    mask = counts < frequency_threshold
    if filter_out:
        mask = ~mask
    return df[mask].copy()


class RareSpecies():
    """Operation to filter rare species, i.e species with a frequency that is below a threshold."""
    def __init__(self, frequency_threshold=10, target_col='species_id', filter_out=True):
        self.frequency_threshold = frequency_threshold
        self.target_col = target_col
        self.filter_out = filter_out

    def __call__(self, df):
        return rare_species(df, self.frequency_threshold, self.target_col, self.filter_out)
