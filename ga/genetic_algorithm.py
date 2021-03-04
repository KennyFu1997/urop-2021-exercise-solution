import numpy as np
from config import CHROMOSOME_LENGTH
from track_generator.command import Command
from track_generator.generator import generate_track
from ga.chromosome_elem import ChromosomeElem

class genetic_computation():
    def __init__(self, CHROMOSOME_LENGTH=CHROMOSOME_LENGTH, num_origin=100000,
                 num_evo=25, cross_prob=0.65, mutation_prob=0.01):
        self.CHROMOSOME_LENGTH = CHROMOSOME_LENGTH
        self.cross_prob = cross_prob
        self.mutation_prob = mutation_prob
        self.num_origin = num_origin
        self.num_evo = num_evo
    
    def get_original_generation(self):
        individuals = []
        for _ in range(self.num_origin):
            individual = [ChromosomeElem(command=Command.S, value=np.random.randint(0, 6))]
            for i in range(0, self.CHROMOSOME_LENGTH - 2):
                cmd = np.random.randint(Command.S.value, Command.DY.value + 1)
                if cmd == Command.DY.value:
                    val = np.random.uniform(1, 20)
                else:
                    val = np.random.randint(0, 6)
                for name, member in Command.__members__.items():
                    if member.value == cmd:
                        temp = member
                        break
                gene = ChromosomeElem(command=temp, value=val)
                individual.append(gene)
            individual.append(ChromosomeElem(command=Command.S, value=np.random.randint(0, 6)))
            individuals.append(individual)
            
        return individuals
        
    def eliminate(self, individuals):
        for id, individual in enumerate(individuals):
            flag = 1
            '''
            angle_acc = 0
            '''
            ### DY -> R / L and R / L -> DY / S
            for idx, gene in enumerate(individual):
                if gene.command == Command.DY:
                    '''
                    angle_acc += gene.value
                    if angle_acc >= 360:
                        flag = 0
                        break
                    '''
                    if individual[idx+1].command == Command.DY or individual[idx+1].command == Command.S:
                        flag = 0
                        break
                if gene.command == Command.R or gene.command == Command.L:
                    if individual[idx+1].command == Command.R or individual[idx+1].command == Command.L:
                        flag = 0
                        break
            if not flag:
                del individuals[id]
        
        return individuals
    
    def calculate_fitness(self, individuals):
        fitness = []
        for individual in individuals:
            track = generate_track(chromosome_elements=individual)
            s = np.array([track[0].x, track[0].y])
            e = np.array([track[-1].x, track[-1].y])
            fitness.append(1 / np.linalg.norm(s - e))
        
        return fitness
    
    def roulette_wheel_selection(self, individuals, fitness):
        prob = []
        random_deci = []
        new_pop = []
        population = len(individuals)
        sum_fitness = sum(fitness)
        for _ in range(population):
            new_pop.append([])
        for idx, individual in enumerate(individuals):
            prob.append(fitness[idx] / sum_fitness)
        prob = np.cumsum(prob)
        for _ in range(population):
            random_deci.append(np.random.random())
        random_deci = sorted(random_deci)
        
        random_idx = 0
        pop_idx = 0
        while random_idx < population:
            if random_deci[random_idx] < prob[pop_idx]:
                new_pop[random_idx] = individuals[pop_idx]
                random_idx += 1
            else:
                pop_idx += 1
        # delete empty element in new_pop
        for i, elem in enumerate(new_pop):
            if elem == []:
                del new_pop[i]
                
        return new_pop
    
    def cross_over(self, parents):
        children = []
        population = len(parents)
        parity = population % 2
        for i in range(0, population - 1, 2):
            temp_father = parents[i]
            temp_mother = parents[i+1]
            cross_point = np.random.randint(0, self.CHROMOSOME_LENGTH-1)
            # cross-over
            # not randomly choose cross-over point
            if np.random.random() < self.cross_prob:
                for j in range(1, self.CHROMOSOME_LENGTH-1):
                    if (temp_father[j].command == Command.DY and temp_mother[j].command == Command.S) \
                        or (temp_father[j].command == Command.S and temp_mother[j].command == Command.DY):
                        temp = temp_father[j:]
                        temp_father[j:] = temp_mother[j:]
                        temp_mother[j:] = temp
                children.append(temp_father)
                children.append(temp_mother)
        if parity == 1:
            children.append(parents[-1])
        
        return children
        
    def mutation(self, children):
        new_population = []
        for child in children:
            if np.random.random() < self.mutation_prob:
                mutation_point = np.random.randint(0, self.CHROMOSOME_LENGTH)
                if child[mutation_point].command == Command.DY:
                    child[mutation_point].value = np.random.uniform(1, 20)
                elif child[mutation_point].command == Command.S:
                    child[mutation_point].value = np.random.randint(0, 6)
                else:
                    child[mutation_point].value = np.random.randint(0, 6)
            new_population.append(child)
             
        return new_population
    
    def evolution(self):
        origin = self.get_original_generation()
        parents = origin
        parents = self.eliminate(parents)
        for _ in range(self.num_evo):
            fitness = self.calculate_fitness(parents)
            selected_parents = self.roulette_wheel_selection(parents, fitness)
            children = self.cross_over(selected_parents)
            children = self.mutation(children)
            parents = children
        final_generation = parents
        
        return final_generation
        
        
                    
            
        
        

