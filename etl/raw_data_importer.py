import pandas as pd
import numpy as np
from ruamel.yaml import YAML
import sys
import argparse

program_version = '0.1'

# Parse command line arguments
parser = argparse.ArgumentParser(description='Parse command line input strings.')

parser.add_argument('-d', action="store", dest='excel_file_path')
parser.add_argument('-c', action="store", default='import_control_file.yml', dest="control_file")
parser.add_argument('-o', action="store", dest="output_directory")
parser.add_argument('--version', action='version', version='%(prog)s '+program_version)

CLI_args = parser.parse_args()

def import_control_dict(control_file):
    """ Import the "import_control_file.yml" to a python dictionary.
    This imports a YAML file in the appropriate format (probably the "import_control_file.yml")
    and returns a python dictionary of that file.

    The dictionary will have the following structure:
    <sheet_name_in_excel_file>:
     output_object_name: <what_your_output_object_will_be_named>
     import_fields:
         <original_field_name>:
             new_field_name: <string_of_new_field_name>
             dtype: <pandas_acceptable_dtype_string>
    """
    yaml = YAML(typ='safe', pure=True)
    yaml.preserve_quotes=True
    return yaml.load(open(control_file, 'r'))

def get_sheet_names(control_dict):
    """ Returns a python array listing the sheet names of the incoming file.
    """
    return [sheet for sheet in control_dict.keys()]

def get_output_name_map(control_dict):
    sheet_names = get_sheet_names(control_dict)
    out_names = [control_dict[sheet]['output_object_name'] for sheet in sheet_names]
    return dict(zip(sheet_names, out_names))

def get_import_cols(sheet, control_dict):
    """ Returns an array set of columns that you want to import.
    args:
        - sheet: string. The sheet in the excel file you want fields for.
        - control_dict: The controluration file that specifies which columns for which sheets.
    returns: An array of column names.
    """
    return control_dict[sheet]['import_fields'].keys()

def get_old_new_field_name_map(sheet, control_dict):
    """ Create a map of old field names to new field names.
     args:
         - sheet: The sheet for which you want to map field names.
         - control_file: The control file that specfies the field mapping.
     returns: a zip object of (original_field_name, new_field_name).
    """
    new_fields = [control_dict[sheet]['import_fields'][field]['new_field_name'] for field in get_import_cols(sheet, control_dict)]
    orig_fields = get_import_cols(sheet, control_dict)
    return dict(zip(orig_fields, new_fields))

def get_old_name_dtype_map(sheet, control_dict):
    """ Create a map of old field names to pandas-friendly data type strings.
    args:
        - sheet: The sheet in the excel file that you are dealing with.
        - control_dict: the dictionary that specifies the old names and the dtypes.
    returns: a zip object of (origin_field_name, dtype).
    """
    field_names = get_import_cols(sheet, control_dict)
    new_dtypes = [control_dict[sheet]['import_fields'][field]['dtype'] for field in field_names]
    return dict(zip(field_names, new_dtypes))

def extract_from_xl_sheet(excel_file_path, control_dict, sheet):

    _usecols=get_import_cols(sheet, control_dict)
    _dtype=get_old_name_dtype_map(sheet, control_dict)

    out_df = pd.read_excel(io=excel_file_path,
                           sheet_name=sheet,
                           usecols=_usecols,
                           dtype=_dtype
                          )

    out_df.rename(mapper=get_old_new_field_name_map(sheet, control_dict), axis=1, inplace=True)

    return out_df

def write_to_pickle(object, output_object_name, directory_destination = '.'):
    object.to_pickle(path=directory_destination+'/'+output_object_name+'.pkl')

def main(excel_file_path, control_file, output_directory):

    control_dict = import_control_dict(control_file)
    sheet_names = get_sheet_names(control_dict)
    out_file_names = get_output_name_map(control_dict)

    for sheet in sheet_names:
        out_object = extract_from_xl_sheet(excel_file_path, control_dict, sheet)
        write_to_pickle(out_object, out_file_names[sheet], directory_destination=output_directory)

if __name__ == '__main__':
    main(CLI_args.excel_file_path, CLI_args.control_file, CLI_args.output_directory)
