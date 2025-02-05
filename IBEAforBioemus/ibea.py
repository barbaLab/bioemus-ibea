""" Indicator-based Evolutionary Algorithm with Epsilon indicator
is an evolutionary algorithm for searching a multi-objective space by improving on the Pareto front.
Formally, the algorithm is classified as a (\mu/\rho + \lambda)-ES, i.e.
\lambda offsprings are produced by \rho mates selected from \mu after a binary tournament.
An Evolution Strategy applies `+` selection if it only takes fitness into account for the selection phase.
"""
from __future__ import division

from collections import deque
import os

from numpy import array, full, min, minimum, maximum, clip, max
from numpy import sqrt, exp, seterr, copy, inf, empty, nan, zeros
import numpy as np
from numpy.random import rand, seed, choice, binomial, randn

from IBEAforBioemus.crossover import bounded_sbx
from IBEAforBioemus.mutation import derandomized_mutation, search_path_mutation, isotropic_mutation

seterr(all='raise')


class IBEA(object):
    def __init__(self, kappa=0.05, alpha=100, n_offspring=25, seedit=42,
                 pr_x=1, pr_mut=0.2, var=5.0, max_generations=200,
                 n_sbx=5, area_max=1, mutation_operator='isotropic'):
        """
        Algorithm parameters

        :param kappa: Fitness scaling ratio
        :param alpha: Population size
        :param n_offspring: Number of offspring individuals
        :param seedit:
        :param pr_x: Crossover probability
        :param pr_mut: Mutation probability
        :param var: Noise level for mutation
        :param max_generations:
        :param n_sbx: Simulated Binary Crossover distribution index - Can be [2, 20], typically {2, 5}
        :param mutation_operator:
        """
        self.kappa = kappa
        self.alpha = alpha
        self.pr_crossover = pr_x
        self.pr_mutation = pr_mut
        self.sigma_init = var  # Vector of std deviation of each paramater to be adjusted: used in mutation step
        self.n_offspring = n_offspring
        self._min = None  # Objective function minima
        self._max = None  # Objective function maxima
        self.indicator_max = None  # Indicator function maximum
        self.max_generations = max_generations
        self.n_sbx = n_sbx
        self.area_max = area_max

        # Choose mutation operator function 
        if mutation_operator == 'derandomized':
            self.mutation_operator = derandomized_mutation
        elif mutation_operator == 'isotropic':
            self.mutation_operation = isotropic_mutation
        elif mutation_operator == 'search_path':
            self.mutation_operation = search_path_mutation
        else:
            raise ValueError  # Something something is not implemented (CMA is it you?)
        self.mutation_op_str = mutation_operator

        # --- Data structure containing: population vectors, fitness and objective values
        self.pop_data = dict()
        # --- Free indices for the population dictionary
        self.free_indices = deque()
        # --- Population counter
        self.population_size = 0

        # --- Random state
        seed(seedit)

    def __str__(self):
        # return 'Indicator-based Evolutionary Algorithm with Epsilon indicator'
        desc = 'ibea_pop{}_offs{}_{}_mut{}_recomb{}_var{}_sbx{}_max_gen{}' \
            .format(self.alpha, self.n_offspring, self.mutation_op_str,
                    self.pr_mutation, self.pr_crossover,
                    self.sigma_init, self.n_sbx, self.max_generations)
        return desc

    def ibea(self, fun, lbounds, ubounds, remaining_budget, goal_distr):
        
        print("Starting optimization:")
        lbounds, ubounds = array(lbounds), array(ubounds)  # [-100, 100]
        dim, f_min = len(lbounds), None
        # dim_sqrt = sqrt(dim + 1)
        # Pr(mutation) = 1/n
        # self.pr_mutation = 1/dim
        # sigma = full(dim,self.sigma_init) # in case sigma is equal for all parameters
        sigma = self.sigma_init
        sigma_count = []
        # 1. Initial population of size alpha
        # Sampled from the uniform distribution [lbounds, ubounds]
        particles = rand(self.alpha, dim) * (ubounds - lbounds) + lbounds
        # particles = randn(self.alpha, dim) * self.sigma_init
        # Rescaling objective values to [0,1]
        tmp = [fun(x, goal_distr, str(i), "genesis") for i, x in enumerate(particles)]
        objective_values_not_scaled, true_distr = zip(*tmp)
        objective_values_not_scaled = array(objective_values_not_scaled)
        true_distr = array(true_distr)
        remaining_budget -= self.alpha
        # Datastructure containing all the population info
        self.pop_data = {
            p: {
                'x': particles[p],
                'obj': 0.0,
                'obj_not_scaled': objective_values_not_scaled[p],
                'true_distribution': true_distr[p],
                'fitness': 0.0,
                'z': empty(dim) * nan # value to update mutation sigma
            } for p in range(self.alpha)
            }
        
        
        # --- Initialize variables
        done = False
        generation = 0
        self.population_size = self.alpha
        # self.free_indices = deque(range(self.alpha, self.alpha + 2 * self.n_offspring))
        print(f"Terminated parent generation, starting recombination\n")
        # search_path = zeros(dim)
        
        obj_monitor = []
        fit_monitor = []
        
        while not done:
            self.free_indices = deque(range(self.alpha + 2 * self.n_offspring * generation, self.alpha + 2 * self.n_offspring * (generation+1)))

            # Rescaling objectives
            all_obj_not_scaled = array([data['obj_not_scaled'] for data in self.pop_data.values()])
            all_obj = self.rescale(all_obj_not_scaled)
            for i, key in enumerate(self.pop_data.keys()):
                self.pop_data[key]['obj'] = all_obj[i]

            # Lazy compute max absolute value of pairwise epsilon indicator 
            key_list = list(self.pop_data.keys())
            self.indicator_max = max([abs(self.epsilon_indicator(i1, i2))
                                  for i1 in key_list
                                  for i2 in key_list
                                  if i1 != i2])

            # 2. Fitness assignment
            for i in self.pop_data.keys():
                self.compute_set_fitness(i)

            fit_all = -inf
            area_monitor = []
            for (item, data) in self.pop_data.items():
                if data['fitness'] >= fit_all:
                    fit_all = data['fitness']
                    obj_sum = data['obj_not_scaled']
                if np.all(data['obj_not_scaled']<1) and np.all(data['obj_not_scaled']>0):
                    area_monitor.append(compute_total_area(data['obj_not_scaled'])) 
            fit_monitor.append(fit_all)
            obj_sum = array(obj_sum)
            obj_monitor.append(obj_sum)
            area_monitor = array(area_monitor)
            # 3 Environmental selection
            env_selection_ok = False
            while not env_selection_ok:
                # 3.1 Environmental selection
                fit_min = inf
                worst_fit = None
                for (k, v) in self.pop_data.items():
                    if v['fitness'] <= fit_min:
                        fit_min = v['fitness']
                        worst_fit = k

                # 3.3 Update fitness values
                for i in self.pop_data.keys():
                    self.pop_data[i]['fitness'] += exp(- self.epsilon_indicator(worst_fit, i) / (self.indicator_max * self.kappa + np.finfo(float).eps))
                # 3.2 Remove individual
                self.population_size -= 1
                self.pop_data.pop(worst_fit)
                self.free_indices.append(worst_fit)
                # Continue while P does not exceed alpha
                env_selection_ok = self.population_size <= self.alpha
                # Removing files with worst metrices
                data_to_remove = f"/home/ubuntu/bioemus/data/raster_genesis{self.free_indices[-1]}.bin"
                sw_to_remove = f"/home/ubuntu/bioemus/config/swconfig_genesis{self.free_indices[-1]}.json"
                hw_to_remove = f"/home/ubuntu/bioemus/config/hwconfig_genesis{self.free_indices[-1]}.txt"

                if os.path.exists(data_to_remove):
                    os.remove(data_to_remove)
                    os.remove(sw_to_remove)
                    os.remove(hw_to_remove)
                
            # Search path mutation
            # if generation > 0:
            #     local_mutations = array([particle['z'] for particle in self.pop_data.values()])
            #     sigma, search_path = search_path_mutation(sigma, local_mutations, dim, self.alpha, search_path)
            #     sigma_count.append(sigma)
            #     for particle in self.pop_data.values():
            #         particle['z'] = empty(dim) * nan
            # 4. Check convergence condition

            done = remaining_budget <= self.alpha + 2 * self.n_offspring or generation >= self.max_generations or np.any(area_monitor >= self.area_max)
            if done: print("OPTIMIZATION TERMINATED"); break
            # 5. Mating selection
            # Perform binary tournament selection with replacement on P in order
            # to fill the temporary mating pool P'.
            item_keys = list(self.pop_data.keys())
            pool = []
            for i in range(2 * self.n_offspring):
                p1, p2 = choice(item_keys, 2, False) #THE SAME INDIVIDUAL CAN GO TWICE IN THE POOL, BECAUSE THE RANDOM CHOICE IS ALWAYS ON THE WHOLE POPULATION
                if self.pop_data[p1]['fitness'] >= self.pop_data[p2]['fitness']:
                    pool.append(p1)
                else:
                    pool.append(p2)
            # 6. Recombination and Variation applied to the mating pool.
            pool_pos = 0
            for i in range(self.n_offspring):
                count = 0

                print(f"\n\tProcessing couple of childrens {i+1}")
                parent1 = self.pop_data[pool[pool_pos]]
                 # HERE IT TAKES COUPLE OF TWO PARENTS, TAKING THE SECOND AS THE FIRST OF THE NEXT ROUND
                parent2 = self.pop_data[pool[pool_pos+1]]
                

                flag = 0
                # Recombination (crossover) operator
                if binomial(1, self.pr_crossover):
                    child1, child2,  flag = bounded_sbx(parent1['x'], parent2['x'],
                                                 lbounds, ubounds, self.n_sbx)
                else:
                    child1 = parent1['x'].copy()
                    child2 = parent2['x'].copy()

                if binomial(1, self.pr_mutation):
                    # assert all(sigma > 0), 'Dirac detected, Variance = {})'.format(sigma)
                    # child1, sigma = derandomized_mutation(child1, sigma, dim)
                    # child2, sigma = derandomized_mutation(child2, sigma, dim)
                    # (Isotropic) mutation
                    child1, child2, z1, z2 = isotropic_mutation(child1, child2, sigma, dim)
                    
                else:
                    z1 = empty(dim) * nan
                    z2 = empty(dim) * nan
                    flag += 1                 

                if flag < len(child1)+1: ## MEANS THAT AT LEAST ONE INDIVIDUAL DIFFERS FROM THE PARENT.

                    # Make sure vectors are still bounded
                    child1 = clip(child1, lbounds, ubounds)
                    child2 = clip(child2, lbounds, ubounds)

                    name = "genesis"
                    obj_c1_not_scaled, true_distribution1 = fun(child1, goal_distr, str(self.free_indices[count]), name)
                    count += 1
                    obj_c2_not_scaled, true_distribution2 = fun(child2, goal_distr, str(self.free_indices[count]), name)
                    count += 1
                    remaining_budget -= 2

                    indx = self.add_offspring(child1, obj_c1_not_scaled, true_distribution1, z1)
                    indx = self.add_offspring(child2, obj_c2_not_scaled, true_distribution2, z2)
                else:
                    indx1 = self.add_offspring(parent1['x'], parent1['obj_not_scaled'], parent1['true_distribution'], parent1['z'])
                    indx2 = self.add_offspring(parent2['x'],  parent2['obj_not_scaled'], parent2['true_distribution'], parent2['z'])
                    print(f"\tAdded children {indx1} identical to parent {pool[pool_pos]}")
                    print(f"\tAdded children {indx2} identical to parent {pool[pool_pos+1]}")
                
                pool_pos+=2
                
            generation += 1
            print(f"Terminated generation {generation}\n")
            
        # Choose vector maximizing fitness
        best_fitness = -inf
        best_fit = None
        for (item, data) in self.pop_data.items():
            if data['fitness'] >= best_fitness:
                best_fitness = data['fitness']
                best_fit = item
        # best = [self.pop_data[p]['x'] for p in self.pop_data.keys()]
        return   best_fit, self.pop_data[best_fit]['x'], self.pop_data[best_fit]['obj_not_scaled'], self.pop_data[best_fit]['true_distribution'], obj_monitor, fit_monitor

    def add_offspring(self, vector, objective_value_not_scaled, true_distribution, z, objective_value = 0.0, fitness=0.0):
        # Add the resulting offspring to P                
        self.population_size += 1
        indx = self.free_indices.popleft()
        self.pop_data[indx] = {
            'x': vector,
            'obj': objective_value,
            'obj_not_scaled': objective_value_not_scaled,
            'true_distribution': true_distribution,
            'fitness': fitness,
            'z': z
        }
        return indx

    def compute_fitness(self, objective_value, exclude=-1):
        """ For all vectors in P\{exclude}, compute pairwise indicator function
        and sum to get fitness value."""
        exp_sum = 0.0
        for (indx, data) in self.pop_data.items():
            if indx != exclude:
                exp_sum -= exp(- compute_epsilon(data['obj'], objective_value)
                               / (self.indicator_max * self.kappa + np.finfo(float).eps))
        return exp_sum

    def compute_set_fitness(self, particle):
        particle_obj = self.pop_data[particle]['obj']
        fitness = self.compute_fitness(particle_obj, particle)
        self.pop_data[particle]['fitness'] = fitness

    def epsilon_indicator(self, i1, i2):
        obj1 = self.pop_data[i1]['obj']
        obj2 = self.pop_data[i2]['obj']
        return compute_epsilon(obj1, obj2)

    def update_max_indicator(self, added_obj):
        epsilons = array([abs(compute_epsilon(x['obj'], added_obj)) for x in self.pop_data.values()])
        self.indicator_max = max([self.indicator_max, epsilons.max()])

    def rescale(self, objective_not_scaled):
        # Save objective lower and upper bounds
        self._min = objective_not_scaled.min(axis=0)
        self._max = objective_not_scaled.max(axis=0)

        # Column-wise rescaling 
        _, ndims = objective_not_scaled.shape
        objective = copy(objective_not_scaled)
        for dim in range(ndims):
            objective[:, dim] = (objective_not_scaled[:, dim] - self._min[dim]) / (self._max[dim] - self._min[dim] + np.finfo(float).eps)
        return objective

    def rescale_one(self, objective_not_scaled):
        # Update objective lower and upper bounds
        self._min = minimum(self._min, objective_not_scaled)
        self._max = maximum(self._max, objective_not_scaled)
        # Rescale vector
        objective = copy(objective_not_scaled)
        for dim in range(objective_not_scaled.shape[0]):
            objective[dim] = (objective_not_scaled[dim] - self._min[dim]) / (self._max[dim] - self._min[dim] + np.finfo(float).eps)
        return objective


def compute_epsilon(obj1, obj2):
    """ Smallest epsilon such that f_i(x1) - f_i(x2) < eps for all i"""
    eps = max(obj1-obj2)
    #assert -1 <= eps <= 1, 'Bounds not respected: O1 = {}, O2 = {}, eps = {}'.format(obj1, obj2, eps)
    return eps

def compute_triangle_area(r1, theta1, r2, theta2):
    return 0.5 * np.abs(r1 * r2 * np.sin(theta2 - theta1))

def compute_total_area(distances):
    # distances with same length as number of verteces
    distances = 1-distances  
    distances = distances.tolist()
    # Calculate total area of radar plot (polygon area)
    angles = np.linspace(0, 2 * np.pi, len(distances), endpoint=False)
    total_area = 0
    for i in range(len(distances) - 1):
        total_area += compute_triangle_area(distances[i], angles[i], distances[i + 1], angles[i + 1])
    # Add area for the last triangle, connecting the last point back to the first
    total_area += compute_triangle_area(distances[-1], angles[-1], distances[0], angles[0])

    return total_area
