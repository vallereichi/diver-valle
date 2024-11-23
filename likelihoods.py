import numpy as np

def example(param:float, mu:float = 50, sigma:float = 20) -> float:
    return -0.5 * np.log(2*np.pi*sigma**2) - (1/(2*sigma**2)) * (param - mu)**2

def combined_log_likelihood(likelihoods:list[float]) -> float:
    return np.sum(likelihoods)


