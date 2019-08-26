import pandas as pd


# Find the distance between two shelters in order to demote long moves and promote shorter moves
def distance_between_shelters(shelter_a, shelter_b):
    # Use shelter_a.location and shelter_b.location
    return 100

""" Determine the adoptability of different categories of animals per shelter. Look at:
 - Age of the animal
 - Image-derived information (breed, hair type, color, size -- can we tell the size of an animal using a CNN? Otherwise can use stats from the breed? Or weight if we have it.)

 Combine this with:
 - Length of stay at the shelter 
 - Outcome of the animal (adopted vs. euthanized vs. died vs ...) If an animal was euthanized, perhaps this can be equivalent to a very, very long stay for the purposes of the model.

 NOTE: if there is a small sample size at a shelter, we need to take this into account. We don't want to be drawing conclusions based on 1 animal. Could even do some sort of automate p-value or something?
"""
def analyze(shelter):
    '''
    Perform an analysis on the shelter to see which animals are most likely to be adopted and which ones are least likely. 
    Analyze based on:
    - age
    - size
    - appearance
    - gender?
    Save the analysis somewhere--perhaps as part of the shelter object?
    '''
    pass

# The fun part!
def recommend(animals, shelters):
    '''
    'animals' is a list of animal objects, and 'shelters' is a list of shelter objects.
    Returns a dict with the keys as the shelters and the values as the animals to be transferred to each shelter

    We also need a mechanism to specify how many animals we are allowed to transfer in to each shelter. Suggestions accepted on the best way to do this?
    '''
    results = {}
    for a in animals:
        pass
    return results



# Take into consideration how old the data is; make the older data less relevant by multiplying it by a factor of 0.3 or something--or a gradient that progressively makes older data less relevant?

if __name__ == '__main__':
    pass
    # create some fake animals and pass to the algorithm to test it