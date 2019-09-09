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
        self.animals = []

        # PERFORM ANALYSIS

    def analyze(self):
        pass
    '''
        0.1, 0.2 
        terriers = 0.3, p-value = 0.99
        rottweilers = .4, p-value = 0.0002

        BUT

        old terriers = 0.8, p-value = 0.001

        while 

        old =.1, p-value=0.001
'''

