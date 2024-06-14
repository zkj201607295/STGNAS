from __future__ import division
import random


class Selection(object):
    def __init__(self, parent_individuals, child_individuals):
        self.individuals = []
        for indi in parent_individuals:
            self.individuals.append(indi)
        for indi in child_individuals:
            self.individuals.append(indi)

    def do_selection(self, args):
        indi_list_len = len(self.individuals)
        assert indi_list_len == int(args.pop_size * 2)
        random_range = []
        while len(random_range) < indi_list_len:
            num = random.randint(0, indi_list_len - 1)
            if num not in random_range:
                random_range.append(num)
        selected_index_list = []
        for i in range(0, indi_list_len, 2):
            if self.individuals[random_range[i]].fitness < self.individuals[random_range[i + 1]].fitness:
                selected_index_list.append(random_range[i])
            else:
                selected_index_list.append(random_range[i + 1])
        next_individuals = [self.individuals[i] for i in selected_index_list]
        return next_individuals






