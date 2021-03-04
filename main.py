import numpy as np
from ga.chromosome_elem import ChromosomeElem
from ga.genetic_algorithm import genetic_computation
from config import CHROMOSOME_LENGTH
from track_generator.command import Command
from track_generator.generator import generate_track

import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    chromosome_elements = [ChromosomeElem(command=Command.S, value=11),
                           ChromosomeElem(command=Command.DY, value=25),
                           ChromosomeElem(command=Command.R, value=9),
                           ChromosomeElem(command=Command.S, value=5)]

    '''
    cross_prob = 0.65
    mutation_prob = 0.01
    num_evo = 25
    num_origin = 100000
    computation = genetic_computation(CHROMOSOME_LENGTH=CHROMOSOME_LENGTH,
                                      num_origin=num_origin,
                                      num_evo=num_evo,
                                      cross_prob=cross_prob,
                                      mutation_prob=mutation_prob)
    candidates = computation.evolution()
    if len(candidates):
        chromosome_elements = candidates[np.random.randint(0, len(candidates))]

        track_points = generate_track(chromosome_elements=chromosome_elements)

        plot_x = [track_point.x for track_point in track_points]
        plot_y = [track_point.y for track_point in track_points]
        plt.scatter(plot_x, plot_y)
        plt.show()
    else:
        print('No off-springs left.')

