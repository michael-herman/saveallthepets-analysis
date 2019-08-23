import pandas as pd 
class shelter(object):
    """
    Class that represents the data belonging to one shelter

    Attributes:
        ...
    """

    def __init__(self, name, path):
        self.name = name
        self.data = pd.read_csv(path) # NOTE: WE DO NOT HAVE THE FINALIZED DATA TEMPLATE
        self.location = None
        


