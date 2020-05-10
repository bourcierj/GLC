"""Defines constants, mostly path to data ressources for the project.
"""
import os

# Read the occurrences_fr_train.csv data
PROJECT_ROOT = os.path.expanduser('~/projects/geolifeclef20/')
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data/')
PATH_OCCURRENCES = os.path.join(DATA_ROOT, 'occurrences/')
PATH_SPECIES_METADATA = os.path.join(PATH_OCCURRENCES, 'species_metadata.csv')
PATH_FR_TRAIN = os.path.join(PATH_OCCURRENCES, 'occurrences_fr_train.csv')
PATH_US_TRAIN = os.path.join(PATH_OCCURRENCES, 'occurrences_us_train.csv')
PATH_FR_TEST = os.path.join(PATH_OCCURRENCES, 'occurrences_fr_test.csv')
PATH_US_TEST = os.path.join(PATH_OCCURRENCES, 'occurrences_us_test.csv')
