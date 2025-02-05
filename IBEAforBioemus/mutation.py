""" Some mutation operators for evolution strategies. """

from __future__ import division

from traceback import format_exc

from numpy import power, abs, multiply, pi
from numpy import seterr, nansum
from numpy import zeros, ones, sqrt, exp
from numpy.linalg import norm
from numpy.random import randn
from math import gamma

seterr(all='raise')

def isotropic_mutation(child1, child2, sigma, dim):
    z1 = randn(dim)
    z2 = randn(dim)
    child1 += z1 * sigma
    child2 += z2 * sigma
    
    return child1, child2, z1, z2


def search_path_mutation(sigma, local_mutations, n_dimensions, mu, search_path):
    """ (mu/mu, lambda)-ES with Search Path Algorithm
        :reference : Nikolaus Hansen, Dirk V. Arnold and Anne Auger,
        Evolution Strategies, February 2015. (Algorithm 4)

        :param sigma: vector of step-sizes and/or standard deviations
        :param local_mutations: matrix of (n_offsprings, n_dimensions)
        sampled from the standard normal distribution
        :param lamda: number of offsprings
        :param mu: number of parents
        :param n_dimensions: dimensions of the search space
        :return: the adapted variance
    """

    # exponentially fading record of mutation steps
    c = sqrt(mu / (n_dimensions + mu))
    d = 1 + sqrt(mu / n_dimensions)
    di = 3 * n_dimensions
    one = ones(n_dimensions)
    multivariate_norm = sqrt(2.0)*gamma((n_dimensions+1)/2)/gamma(n_dimensions/2)
    univariate_norm = sqrt(2.0/pi)
    search_path *= (1 - c)
    search_path += sqrt(c * (2 - c)) * sqrt(mu) / mu * nansum(local_mutations,axis=0)
    sigma_abs = exp((abs(search_path)/univariate_norm - one) / di)
    sigma_norm = exp((norm(search_path)/multivariate_norm - 1) * (c / d))
    sigma = multiply(sigma, sigma_abs) * sigma_norm

    return sigma, search_path
#########################################################################################################################################################
### FROM HERE ALL THE SELF ADAPTED MUTATION STRATEGIES NEED THE EVALUATION OF fun(x) INSIDE TO RANK THE RESULTED MUTATION AND FOLLOW THE RIGHT SEARCHPATH
### (and the algorithms should be revised according to this) 
### NOT WORKING FROM HERE

def derandomized_mutation(x, sigma, n_dimensions):
    """ Adaptation of the variance with both global and dimension-wise components.
        :sigma : vector of step-sizes and/or standard deviations
        :reference : Nikolaus Hansen, Dirk V. Arnold and Anne Auger,
        Evolution Strategies, February 2015. (Algorithm 3)
        """
    try:
        # Initialize variables
        di = n_dimensions
        d = sqrt(n_dimensions)
        tau = 1 / 3
        ksi = tau * randn()
        z = randn(n_dimensions)
        one = ones(n_dimensions)

        # Mutate vector
        xRes = x + exp(ksi) * multiply(sigma, z)
        # Compute sigma adaptation
        adaptation_vect1 = power(exp(abs(z)/sqrt(2.0/pi)-one),1.0/di)
        adaptation_vect2 = power(exp(ksi),1.0/d)
        # Compute new value of sigma
        adapted_sigma = multiply(sigma, adaptation_vect1) * adaptation_vect2
    except FloatingPointError | RuntimeWarning | ValueError:
        print(format_exc())
        exit(2)

    return xRes, adapted_sigma





def one_fifth_update_sigma(sigma, offspring_fitness, parent_fitness, inv_dim_sqrt):
    """ Adapt step-size using the 1/5-th rule, :references [Retchenberg, Schumer, Steiglitz]
    The idea is to raise, in expectation, the log of the variance if the success probability
        is larger than 1/5, and decrease it otherwise. Note: for our fitness function bigger is better
        :param sigma: initial variance
        :param offspring_fitness: fitness value of the child
        :param parent_fitness: fitness value of the parent
        :return: adapted variance
        :param inv_dim_sqrt: inversed of the square root of problem dimension.
        """
    indicator = int(parent_fitness <= offspring_fitness)
    sigma *= power(exp(indicator - 0.2), inv_dim_sqrt)
    return sigma