#!/usr/bin/python3

'''
This script will import the raw data xls file and create a series of pandas tables with select portions of the data.
'''

import sys
from ruamel.yaml import YAML
import numpy as np
import pandas as pd

data_file_path = sys.argv[0]
config_file_path = sys.argv[1]


# TODO: Went a little too far with the config file approach. Need methods that are recyclable and can be used in code but still can be used with the config file.

def AnimalExtract_importer(file_path, config_path):
    ''' Given a file path string, this imports data from the AnimalExtract file.
    '''

    # Get the config file with the field info dictionary.
    yaml = YAML(typ='safe', pure=True)
    yaml.preserve_quotes = True
    config_dict = yaml.load(open('config_import_animal_table.yml', 'r'))

    # Import the data.
    animal_extract = pd.read_excel(io=file_path, sheet_name="AnimalExtract", usecols=config_dict['_AnimalExtract_import_cols'], dtype=config_dict['_AnimalExtract_dtypes'])

    animal_extract.rename(mapper=config_dict['_AnimalExtract_rename_dict'], axis=1, inplace=True)

    return animal_extract

#
def main(data_file_path, config_file_path):

    return AnimalExtract_importer(data_file_path, config_file_path)


if __name__ == '__main__':
    main()
