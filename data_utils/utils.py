import pandas as pd
import torch


def load_occurrences_data(path_csv, reveals_targets=True):
    """Loads the occurrences training data.
    Args:
        path_csv (str): path to the csv file.
        reveals_targets (bool, default: True): bool indicating if the target species are available as a column.
    Returns:
        (pandas.DataFrame): data read
    """
    from constants import PATH_SPECIES_METADATA
    # read the csv
    df = pd.read_csv(path_csv, sep=';', header=0, index_col='id', low_memory=True)
    if reveals_targets:
        # read the species metadata
        df_metadata = pd.read_csv(PATH_SPECIES_METADATA, sep=';', header=0,
                                  index_col='species_id', low_memory=True)
        # merge to get species names
        df = df.merge(
            df_metadata, how='left', left_on='species_id', right_index=True, copy=False
        ).drop('GBIF_species_id', axis='columns')

    return df


if __name__ == '__main__':
    from constants import PATH_FR_TRAIN, PATH_FR_TEST
    train_df = load_occurrences_data(PATH_FR_TRAIN)
    test_df = load_occurrences_data(PATH_FR_TEST, reveals_targets=False)
