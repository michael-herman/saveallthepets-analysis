"""Contains utility functions used across modules"""

import os
from datetime import datetime
from configparser import ConfigParser

# Module constants
ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.abspath(os.path.join(ROOT_PATH, 'data'))
TIMESTAMP = datetime.now().strftime('%m%d%Y')
CONFIG_FILE = os.path.abspath(os.path.join(ROOT_PATH, 'config.ini'))


def config(section: str, filename=CONFIG_FILE,):
    """Returns parameters for given section of the config.ini file.

    Raises ValueError if section not found in config.ini
    """
    parser = ConfigParser()
    parser.read(filename)

    # get section parameters
    pars = dict()
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            pars[param[0]] = param[1]
    else:
        raise ValueError('Section {0} not found in the {1} file'.format(section, filename))

    return pars
