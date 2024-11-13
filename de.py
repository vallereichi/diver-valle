import numpy as np
import random

# Step1: Initialize Generation
def create_base_vector(D:int, ranges:np.array) -> np.array:
    """
    create points in a D-dimensional parameter space

    parameters:
        D: dimension of vector
        ranges: acceptance range in each dimension

    returns:
        array of size D
    """

    X_i = []
    for i in range(D):
        X_i.append(random.uniform(ranges[i][0], ranges[i][1]))
    
    return np.array(X_i)


def initialize_generation(D:int, ranges:np.array, NP:int = None) -> np.array:
    """
    Function to initialize the first generation of base vectors used for the differential evolution.
    Per default it creates each point by random sampling each dimension from from the specified range.

    parameters:
        D: dimensionality of each point
        ranges: acceptance range in each dimension
        NP: number of initial points. defaults to 10*D

    returns:
        array of size NP with points of dimension D
    """

    if NP == None:
        NP = 10 * D

    X = []

    for i in range(NP):
        X.append(create_base_vector(D, ranges))

    X = np.array(X)
    while len(np.unique(X, axis=0)) != len(X):
        X = np.unique(X, axis=0)
        np.append(X, create_base_vector(D, ranges))
        

    print(f"[INFO]: initialized first genration with {NP} unique base vectors of dimension {D}")
    return X



    
# Step2: Mutation
# Step3: Crossover
# Step4: Selection
# Step5: Break conditions