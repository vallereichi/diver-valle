import numpy as np

def example(param:float, mu:float = 0.5, sigma:float = 0.5) -> float:
    return -0.5 * np.log(2*np.pi*sigma**2) - (1/(2*sigma**2)) * (param - mu)**2

def combined_log_likelihood(likelihoods:list[float]) -> float:
    return np.sum(likelihoods)


