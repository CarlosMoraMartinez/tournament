import numpy as np
import matplotlib.pyplot as plt
import sys


class Organism:
    
    """
    This class will allow you to define the representation of an object (i.e., a vector of parameters or a more flexible implementation)
    """
    
    def __init__(self, num_genes, genes = None):
        if(genes is None):
            #self.chromosome = np.random.rand(num_genes).tolist()
            self.chromosome = np.random.uniform(size=num_genes, low = 0, high = 20).tolist()
        else:
            self.chromosome = genes
    def __getitem__(self, i):
        return self.chromosome[i]

    
class StopCriteria:
    
    """
    Modifying this class will allow you to implement flexible ways of stopping a simulation, taking various things into account
    In this case, it stops if the mean squared error of a population compared to the best possible value
    (in this case we know the optimal) is smaller than a threshold
    """

    def __init__(self, number, threshold):
        self.optimal = number
        self.threshold = threshold
    def compare(self, fitness):
        return np.sum(np.power(self.optimal - fitness, 2))/fitness.shape[0]
    def stop(self, fitness):
        if(self.compare(fitness) < self.threshold):
            return True
        else:
            return False
        
class StopCriteria2(StopCriteria):
    
    """
    It stops when the error is < than a threshold
    """
    
    def __init__(self, number, threshold):
        self.optimal = number #0 in thi version
        self.threshold = threshold
    def stop(self, fitness):
        mm = np.mean(fitness)
        if(mm == 0):
            return False
        compare = 1/mm
        if(abs(compare) < self.threshold):
            return True
        else:
            return False


class TournamentPopulation:
    
    """
    This class contains the core of the tournament selection algorithm.
    Holds a population of customized objects (Organisms) and applies a function (simulation) to them
    in order to get their fitness.
    The fittest organisms reproduce sexually
    """
    
    def __init__(self, N, K, num_genes, mutation_rate, max_generations=100000):
        self.N = N #Number of individuals in a population
        self.K = K #Selection is stronger when this parameter is higher
        self.num_genes = num_genes #Number of parameters in an individual
        self.mutation_rate = mutation_rate  #You can adjust this dynamically; low mutation rate in the end of the simulation will allow you to fine-tune parameters
        self.max_generations = max_generations
        self.generation = 0
        self.individuals = [Organism(num_genes) for i in range(N)] 
        self.current = 0
    def __getitem__(self, i):
        return self.individuals[i]
    def __iter__(self):
        self.current = 0
        return self
    def __next__(self):
        if(self.current < self.N):
            ind = self[self.current]
            self.current += 1
            return ind
        else:
            raise StopIteration()
    def tournamentSelection(self, simulation, stopCriteria):
        
        """
        Tournament selection algorithm
        ARGUMENTS:
        simulation: any function that gets an Organism object and returns its fitness value (fitness: is higher for better organisms, it can be calculated as 1/error)
        stopCriteria: an object with a method called "stop", which receives the vector of population fitnesses, and has information enough to decide whether the simulation has to stop.

        RETURNS:
        Mean fitness of the population
        """
        
        fitness = np.zeros(self.num_genes)
        mean_fitness = [np.mean(fitness)]
        while(not stopCriteria.stop(fitness) and self.generation < self.max_generations):
            fitness = np.array([simulation(individual.chromosome) for individual in self])  ## This step can be easily parallelized with multiprocessing library
            next_generation = [self.getNewIndividual(fitness) for i in range(self.N)]
            self.individuals = next_generation
            mean_fitness.append(np.mean(fitness))
            self.generation += 1
            if(self.generation %1000 == 0):
                print(self.generation, ': ', mean_fitness[-1])
        return mean_fitness
    def getNewIndividual(self, fitness):
        participants = np.random.choice(range(self.N), self.K, replace = False) #Select K random individuals
        parent_index1 = participants[np.argmax(fitness[participants])] #The best one will be parent 1
        participants = np.random.choice(range(self.N), self.K, replace = False) #The same for parent 2
        parent_index2 = participants[np.argmax(fitness[participants])]
        return self.combine2individuals(self[parent_index1], self[parent_index2]) #Mix individuals
    def combine2individuals(self, org1, org2):
        #prop = [np.mean([org1[gene], org2[gene]])for gene in range(self.num_genes)]
        new_chromosome = [np.random.choice([org1[gene], org2[gene]]) + np.random.uniform(low=-1*self.mutation_rate, high=self.mutation_rate) for gene in range(self.num_genes)]
        #new_chromosome = prop #mean instead of recombination
        new_chromosome = [i if i > 0 else 0.1 for i in new_chromosome] #restrict values
        return Organism(len(new_chromosome), new_chromosome)



def simulationTestGaussian(params):

    """
    Test function 1: gaussian with mean = 10, sigma = 2.5
    The individuals will try to guess what is the mean of the gaussian
    You can define any function and pass it to the tournamentSelection method of a TournamentPopulation object
    """

    mu = 0
    sigma = 2.5
    val = np.mean(params) # Each individual has several genes, here we will use the mean of all of them. In a real simulation each gene would be a different parameter
    a = 1/(sigma*np.pi)
    b = - 0.5 * np.power((mu - val)/sigma, 2)
    return a*np.exp(b)

def gaussian(mu, sigma, start, end):

    """
    This function returns 100 values of a gaussian function with customized mu and sigma, between x = start and x = end
    """
    
    val = np.linspace(start, end, 100)
    a = 1/(sigma*np.pi)
    b = - 0.5 * np.power((mu - val)/sigma, 2)
    return a*np.exp(b)

def optimalGaussian(mu, sigma):
    
    """
    Returns a function that calculates error of given gaussian with respect to the target/optimal gaussian
    """
    
    optimal = gaussian(mu, sigma, mu-3.5*sigma, mu+3.5*sigma) #Calculate values of optimal gaussian
    def simulationTestGaussian2(params):
        """
        Compare a gaussian with optimal gaussian
        """
        x = gaussian(params[0], params[1], mu-3.5*sigma, mu+3.5*sigma)
        error = np.sum(np.power(optimal - x, 2))/optimal.shape[0]
        return 1/error
    return simulationTestGaussian2

def main():
    """
    Arguments:
    - Max error (recommended: 1e-8 to 1e-10; bigger error for smaller sigma values)
    - Mean of Optimal Gaussian (recommended: <=20)
    - Sigma of Optimal Gaussian (recommended: 3)
    - Mutation rate (recommended: around 0.01)
    - Population size N (recommended: 100)
    - K parameter for tournament selection (recommended: 40 for N = 100)
    """
    #optimal = simulationTestGaussian([10])
    optimal = 0
    if(len(sys.argv) < 7):
        max_error = 1e-10  #works with 1e-8 to 1e-10 (for small sigma, use higher error)
        mu = 10  #Works with mu <=20
        sigma = 3
        mutation_rate = 0.01  #works with 0.25
        N = 100
        K = 40
    else:
        max_error = sys.argv[1]
        mu = sys.argv[2]
        sigma = sys.argv[3]
        mutation_rate = sys.argv[4]
        N = sys.argv[5]
        K = sys.argv[6]
    stop = StopCriteria2(optimal, max_error)
    sim_function = optimalGaussian(mu, sigma)
    pop = TournamentPopulation(N = N, K = K, num_genes = 2, mutation_rate = mutation_rate)
    time_fitness = np.log10(pop.tournamentSelection(sim_function, stop))
    #Show the guess of the best individual
    #a = [(10 - np.mean(p.chromosome),np.mean(p.chromosome))  for p in pop]
    a = [(p[0] , p[1], np.mean([abs(10 - p[0])/10, abs(2.5 - p[1])/2.5]))  for p in pop]
    b = sorted(a, key=lambda x: x[2])[0]
    s="Tried to guess the parameters of a gaussian with MU = {} and SIGMA = {}. This is the best individual's guess: mu = {}, sigma = {}.".format(mu, sigma, round(b[0], 2), round(b[1], 2))
    print(s)
    plt.plot(range(len(time_fitness)), time_fitness)
    plt.xlabel("generation")
    plt.ylabel("mean population fitness")
    s="MU = {}, SIGMA = {}. Evolved mu = {}, sigma = {}.".format(mu, sigma, round(b[0], 2), round(b[1], 2))
    plt.title(s)
    #plt.hlines(optimal, 0, len(time_fitness), label = "optimal value")
    plt.show()

    

if __name__== "__main__":
  main()
