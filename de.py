import numpy as np
import random
import yaml
from yaml import Loader
import likelihoods
from likelihoods import combined_log_likelihood


def likelihood(vector:np.array, params:list[float]) -> float:
    """
    calculates the log-likelihood of a given input vector

    parameters:
        vector:  one dimensional input array of arbitrary length
        params: expectation value and width of probability density

    returns:
        calculated log-likelihood of the input vector as a float
    """
    if len(params) != 2:
        return "[ERROR]: length of parameter list must be 2"

    mu = params[0]
    sigma = params[1]

    try:
        D = len(vector)
        Sum = np.sum([(vector[i] - mu) ** 2 for i in range(D)])
    except:
        D = 1
        Sum = (vector - mu) ** 2

    return -(0.5 * D) * np.log(2 * np.pi * sigma**2) - (1/(2* sigma**2)) * Sum


def calculate_likelihood(vector:np.array, capabilities:list[str]) -> dict:
    likes = []
    par_likes = []
    capability_functions = [getattr(likelihoods, capabilities[i]) for i in range(len(capabilities))]

    for i in range(len(vector)):
        for j in range(len(capability_functions)):
            likes.append(capability_functions[j](vector[i][j]))
        par_likes.append(np.array(likes))
        likes = []


    return np.array(par_likes)


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
        
    
    print(f"[INFO]: initialized first generation with {NP} unique base vectors of dimension {D}")
    return X



    
# Step2: Mutation
def mutation(current_generation:np.array, F:float) -> np.array:
    """
    performs the mutation step of the differential evolution by adding a scaled difference vector
    to a base base vector.

    parameters:
        current_generation: array of points with the currrent best likelihood
        F: scaling factor for the difference vector (range: 0 <= F <= 1)

    reuturns:
        array of size NP containing the donor vectors
    """
    if F <= 0 or F >= 1:
        return "[ERROR]: scaling factor F is out of range"
    
    donor_vectors = []

    for i in range(len(current_generation)):
        base_vectors = random.sample(list(current_generation), 3)

        while np.array_equal(current_generation[i], base_vectors[0]) or \
              np.array_equal(current_generation[i], base_vectors[1]) or \
              np.array_equal(current_generation[i], base_vectors[2]) :
            base_vectors = random.sample(list(current_generation), 3)

        X_1, X_2, X_3 = base_vectors
        donor_vectors.append(X_1 + F * (X_2 - X_3))

    print(f"[INFO]: created array of donor vectors with shape {np.array(donor_vectors).shape}")
    
    return np.array(donor_vectors)
        




# Step3: Crossover
def crossover(current_generation:np.array, donor_vectors:np.array, Cr:float) -> np.array:
    """
    performs the crossover step of the differential evolution. This step creates a set of trial vectors for the next generation.

    parameters:
        current_generation: array of size NP containing the current vectors
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
                U_i.append(current_generation[i][j])
            else:
                return "[ERROR]: an unknown error occurred"
            
        
        l = random.randint(0, len(U_i) - 1)
        U_i[l] = donor_vectors[i][l]
        trial_vectors.append(U_i)

    print(f"[INFO]: created array of trial vectors with shape {np.array(trial_vectors).shape}")
    
    return np.array(trial_vectors)

    


# Step4: Selection
def selection(current_generation:np.array, trial_vectors:np.array, current_likelihoods:np.array, trial_likelihoods:np.array) -> np.array:
    """
    performs the selection step of the differential evolution. 
    Since DE is a greedy algorithm the new generation only accepts vectors with a better likelihood

    paramters:
        current_generation: array of size NP containing the current vectors
        trial_vectors: array of size NP containing the trial vectors
        likelihood_params: list containing the parameters for the likelihood function

    returns:    
        array of size NP containing the new generation
    """
    new_generation = []
    new_likelihoods =[]
    for i in range(len(current_generation)):
        if abs(combined_log_likelihood(current_likelihoods[i])) < abs(combined_log_likelihood(trial_likelihoods[i])):
            new_generation.append(current_generation[i])
            new_likelihoods.append(current_likelihoods[i])
        if abs(combined_log_likelihood(current_likelihoods[i])) >= abs(combined_log_likelihood(trial_likelihoods[i])):
            new_generation.append(trial_vectors[i])
            new_likelihoods.append(trial_likelihoods[i])

    if len(new_generation) != len(current_generation):
        return "[ERROR]: length of current and new generation is not the same"
    
    print(f"[INFO]: created a new generation of shape {np.array(new_generation).shape} according to the selection process")

    return np.array(new_generation), np.array(new_likelihoods)




# Step5: Break conditions and loop
def read_input(filepath:str):
    """
    This function reads an input yaml file and returns the different sections "parameters" and "diver" as seperate dictionaries.
    The parameter section should contain all parameters for the diver run and the diver section contains general parameters:

        ''' parameters:
                param1:
                    range: [a,b]
                    likelihood: 'parameter specific likelihood'
                param2:
                    .
                    .
                    .
                
            diver:
                NP: 'number of chains to initialize'
                F: 'Scaling factor for Mutation'
                Cr: 'Crossover parameter'
                    .
                    .
                    .
        '''
    """
    stream = open(filepath, 'r')
    dict = yaml.load(stream, Loader=Loader)

    if "NP" not in dict['diver'].keys():
        return "[ERROR]: You need to specify the number of chains in the input yaml file"
    
    for parameter in dict['parameters'].keys():
        if "range" not in dict['parameters'][parameter].keys():
            return f"[ERROR]: No range specified for parameter {parameter}"
        if "capability" not in dict['parameters'][parameter].keys():
            return f"[ERROR]: No capability specified for parameter {parameter}"
    
    return dict['parameters'], dict['diver']






def diver(input_file:str) -> list[np.array]:
    """
    Full run of the differential evolution algorithm

    parameters:
        D: dimension of the vectors
        ranges: acceptance range in each dimension
        F: scaling factor for the difference vector (range: 0 <= F <= 1)
        Cr: Crossover rate (0 <= Cr <= 1)
        NP: number of initial points. defaults to 10 * D
        steps: number of iterations the algorithm will perform. defaults to 10
        convthresh: convergence threshhold as a break condition for the algorithm. defaults to None

    returns:
        list of arrays containing all generations of the differential evolution
    """
    MAX_STEPS = 10000
    counter = 1
    conv_diff = 0
    ranges = []
    capabilities = []

    # handling the input file and setting default parameters
    parameter_dict, diver_dict = read_input(input_file)

    for parameter in parameter_dict.keys():
        ranges.append(parameter_dict[parameter]['range'])
        capabilities.append(parameter_dict[parameter]['capability'])

    D = len(ranges)

    try:
        NP = diver_dict['NP']
    except: pass
    try:
        F = diver_dict['F']
    except: pass
    try:
        Cr = diver_dict['Cr']
    except: pass
    try:
        steps = diver_dict['steps']
    except: pass
    try:
        convthresh = diver_dict['convthresh']
    except: pass

    if "NP" not in diver_dict.keys():
        NP = 10* D
    if "F" not in diver_dict.keys():
        F = 0.8
    if "Cr" not in diver_dict.keys():
        Cr = 0.8
    if "steps" not in diver_dict.keys():
        steps == 10
    if "convthresh" not in diver_dict.keys():
        convthresh = None

    
    
    if convthresh != None:
        steps = MAX_STEPS

    if convthresh == None:
        convthresh = conv_diff

    current_generation = initialize_generation(D, ranges, NP)
    current_likelihoods = calculate_likelihood(current_generation, capabilities)
    output = [current_generation]

    while (counter <= steps) and (convthresh <= conv_diff):
        print(f"[STEP {counter}] running differential evolution")
        donor_vectors = mutation(current_generation, F)
        trial_vectors = crossover(current_generation, donor_vectors, Cr)
        trial_likelihoods = calculate_likelihood(trial_vectors, capabilities)
        next_generation, next_likelihoods = selection(current_generation, trial_vectors, current_likelihoods, trial_likelihoods)

        

        counter += 1
        conv_diff = np.max([combined_log_likelihood(current_likelihoods[i]) - combined_log_likelihood(next_likelihoods[i]) for i in range(len(next_likelihoods))])
        output.append(next_generation)
        current_generation = next_generation
        current_generation = next_likelihoods

        print(f"[INFO]: Likelihood difference: {conv_diff}")

    return output
    





    
