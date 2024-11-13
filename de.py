import numpy as np
import random

# Step1: Initialize Generation
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
    X_i = []

    for i in range(NP):
        for j in range(D):
            X_i.append(random.uniform(ranges[j][0], ranges[j][1]))
        X.append(X_i)
        X_i = []

    print(f"[INFO]: initialized first genration with {NP} points of dimension {D}")
    return X


    
# Step2: Mutation
# Step3: Crossover
# Step4: Selection
# Step5: Break conditions