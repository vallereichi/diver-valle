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


def initialize_generation(D:int, ranges:np.array, NP:int = None) -> list[np.array]:
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
        

    print(f"[INFO]: initialized first generation with {NP} unique base vectors of dimension {D}")
    return list(X)



    
# Step2: Mutation
def mutation(current_population:np.array, F:float) -> np.array:
    """
    performs the mutation step of the differential evolution by adding a scaled difference vector
    to a base base vector.

    parameters:
        current_population: array of points with the currrent best likelihood
        F: scaling factor for the difference vector (range: 0 <= F <= 1)

    reuturns:
        array of size NP containing the donor vectors
    """
    if F <= 0 or F >= 1:
        return "[ERROR]: scaling factor F is out of range"
    
    donor_vectors = []

    for i in range(len(current_population)):
        base_vectors = random.sample(current_population, 3)

        while np.array_equal(current_population[i], base_vectors[0]) or \
              np.array_equal(current_population[i], base_vectors[1]) or \
              np.array_equal(current_population[i], base_vectors[2]) :
            base_vectors = random.sample(current_population, 3)

        X_1, X_2, X_3 = base_vectors
        donor_vectors.append(X_1 + F * (X_2 - X_3))
    
    return np.array(donor_vectors)
        




# Step3: Crossover
def crossover(current_population:np.array, donor_vectors:np.array, Cr:float) -> np.array:
    """
    performs the crossover step of the differential evolution. This step creates a set of trial vectors for the next generation.

    parameters:
        current_population: array of size NP containing the current vectors
        donor_vectors: array of size NP containing the donor vectors
        Cr: Crossover rate (0 <= Cr <= 1)

    returns:
        array of size NP containing the trial vectors
    """
    if Cr <= 0 or Cr >= 1:
        return "[ERROR]: Crossover rate Cr is out of range"


    trial_vectors = []
    for i in range(len(donor_vectors)):
        U_i = []
        for j in range(len(donor_vectors[i])):
            r = random.uniform(0, 1)

            if r <= Cr:
                U_i.append(donor_vectors[i][j])
            elif r > Cr:
                U_i.append(current_population[i][j])
            else:
                return "[ERROR]: an unknown error occurred"
            
        
        l = random.randint(0, len(U_i) - 1)
        U_i[l] = donor_vectors[i][l]
        trial_vectors.append(U_i)
    
    return np.array(trial_vectors)

    


# Step4: Selection
# Step5: Break conditions