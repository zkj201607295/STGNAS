import random
import copy
from population import Individual


class CrossoverAndMutation(object):
    def __init__(self, individuals, args, num_node_features, num_classes, gen_no, Log):
        self.args = args
        self.individuals = individuals
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.gen_no = gen_no
        self.log = Log

    def process(self, genotype_records, ind_records):
        individuals = []
        geno_times = 0
        para_times = 0
        bad_times = 0
        while len(individuals) < self.args.pop_size:
            crossover = Crossover(self.individuals, self.args.pc, self.log)
            offsprings, info = crossover.do_crossover()
            mutation = Mutation(offsprings, info, self.args.pm, self.log)
            offsprings, info = mutation.do_mutation()
            for i, offspring in enumerate(offsprings):
                genotype_record = str(offspring)
                ind_record = genotype_record + str(info[i][0]) + str(info[i][1]) + str(info[i][2]) + str(info[i][3])
                if genotype_record not in genotype_records:
                    genotype_records.append(genotype_record)
                    ind_records.append(ind_record)
                    indi = Individual(self.args, self.num_node_features, self.num_classes)
                    indi.initialize(genotype=offspring, lr=info[i][0], weight_decay=info[i][1], dropout=info[i][2], hidden_dim=info[i][3])
                    individuals.append(indi)
                    geno_times += 1
                elif ind_record not in ind_records:
                    ind_records.append(ind_record)
                    indi = Individual(self.args, self.num_node_features, self.num_classes)
                    indi.initialize(genotype=offspring, lr=info[i][0], weight_decay=info[i][1], dropout=info[i][2], hidden_dim=info[i][3])
                    individuals.append(indi)
                    para_times += 1
                else:
                    bad_times += 1
                    if bad_times > len(self.individuals) // 2:
                        indi = Individual(self.args, self.num_node_features, self.num_classes)
                        indi.initialize(genotype=offspring, lr=info[i][0], weight_decay=info[i][1], dropout=info[i][2], hidden_dim=info[i][3])
                        individuals.append(indi)
        self.log.info('\tFor {}-generation, geno times={}, para times={}, bad times={}'.format(self.gen_no, geno_times, para_times, bad_times))
        return individuals[0:self.args.pop_size], genotype_records, ind_records


class Crossover(object):
    def __init__(self, individuals, prob_, Log):
        self.individuals = individuals
        self.prob = prob_
        self.log = Log

    def _choose_one_parent(self):
        count_ = len(self.individuals)
        idx1, idx2 = random.randint(0, count_ - 1), random.randint(0, count_ - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, count_ - 1)
        if self.individuals[idx1].fitness < self.individuals[idx1].fitness:
            return idx1
        else:
            return idx2

    def _choose_two_diff_parents(self):
        idx1 = self._choose_one_parent()
        idx2 = self._choose_one_parent()
        while idx2 == idx1:
            idx2 = self._choose_one_parent()
        assert idx1 < len(self.individuals)
        assert idx2 < len(self.individuals)
        return idx1, idx2

    def do_crossover(self):
        _stat_param = {'offspring_new': 0, 'offspring_from_parent': 0}
        new_offspring_list = []
        new_offspring_info = []
        ind1, ind2 = self._choose_two_diff_parents()
        parent1, parent2 = copy.deepcopy(self.individuals[ind1].genotype), copy.deepcopy(self.individuals[ind2].genotype)
        paras1 = [self.individuals[ind1].lr, self.individuals[ind1].weight_decay, self.individuals[ind1].dropout, self.individuals[ind1].hidden_dim]
        paras2 = [self.individuals[ind2].lr, self.individuals[ind2].weight_decay, self.individuals[ind2].dropout, self.individuals[ind2].hidden_dim]

        if not isinstance(parent1, list):
            parent1 = parent1.tolist()
        if not isinstance(parent2, list):
            parent2 = parent2.tolist()
        p_ = random.random()
        if p_ < self.prob:
            pos1, pos2 = random.randint(0, len(paras1) - 1), random.randint(0, len(paras1) - 1)
            if pos1 > pos2:
                tmp = pos1
                pos1 = pos2
                pos2 = tmp
            _paras1, _paras2 = [], []
            for i in range(pos1):
                _paras1.append(paras1[i])
                _paras2.append(paras2[i])
            for i in range(pos1, pos2):
                _paras1.append(paras2[i])
                _paras2.append(paras1[i])
            for i in range(pos2, len(paras1)):
                _paras1.append(paras1[i])
                _paras2.append(paras2[i])
            new_offspring_info.append(_paras1)
            new_offspring_info.append(_paras2)
        else:
            new_offspring_info.append(paras1)
            new_offspring_info.append(paras2)

        p_ = random.random()
        if p_ < self.prob:
            parent1_len, parent2_len = len(parent1), len(parent2)
            parent1_pos1, parent1_pos2 = random.randint(0, parent1_len - 1), random.randint(0, parent1_len - 1)
            if parent1_pos1 > parent1_pos2:
                tmp = parent1_pos1
                parent1_pos1 = parent1_pos2
                parent1_pos2 = tmp
            parent2_pos1, parent2_pos2 = random.randint(0, parent2_len - 1), random.randint(0, parent2_len - 1)
            if parent2_pos1 > parent2_pos2:
                tmp = parent2_pos1
                parent2_pos1 = parent2_pos2
                parent2_pos2 = tmp
            offspring1, offspring2 = [], []
            for i in range(parent1_pos1):
                offspring1.append(parent1[i])
            for i in range(parent2_pos1, parent2_pos2):
                offspring1.append(parent2[i])
            for i in range(parent1_pos2, parent1_len):
                offspring1.append(parent1[i])

            for i in range(parent2_pos1):
                offspring2.append(parent2[i])
            for i in range(parent1_pos1, parent1_pos2):
                offspring2.append(parent1[i])
            for i in range(parent2_pos2, parent2_len):
                offspring2.append(parent2[i])

            new_offspring_list.append(offspring1)
            new_offspring_list.append(offspring2)
        else:
            new_offspring_list.append(parent1)
            new_offspring_list.append(parent2)
        return new_offspring_list, new_offspring_info


class Mutation(object):
    def __init__(self, offsprings, info, prob_, Log):
        self.offsprings = offsprings
        self.info = info
        self.prob = prob_
        self.log = Log

    def do_mutation(self):
        for index, offspring in enumerate(self.offsprings):
            if not isinstance(offspring, list):
                offspring = offspring.tolist()
            r_ = random.random()
            if r_ < self.prob:
                if random.random() < 0.5:  # lr
                    choose = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
                    self.info[index][0] = choose[random.randint(0, 9)]
                if random.random() < 0.5:  # weight_dacey
                    choose = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
                    self.info[index][1] = choose[random.randint(0, 5)]
                if random.random() < 0.5:  # dropout
                    self.info[index][2] = round(random.random(), 1)
                    while self.info[index][2] <= 0.0 or self.info[index][2] >= 1.0:
                        self.info[index][2] = round(random.random(), 1)
                if random.random() < 0.5:
                    choose = [16, 32, 64, 128, 256, 512]
                    self.info[index][3] = choose[random.randint(0, 5)]
            r_ = random.random()
            if r_ < self.prob:
                offspring_before_str = ''
                for num in offspring:
                    offspring_before_str += '%d' % num
                offspring_len = len(offspring)
                mutation_type = random.randint(0, 2)
                if mutation_type == 0:
                    status, r2 = False, random.random()
                    if offspring_len > 1:
                        begin = offspring[0]
                        for i in range(1, offspring_len):
                            if offspring[i] != begin:
                                status = True
                                break
                        if status and r2 < 0.5:  # exchange
                            pos1, pos2 = random.randint(0, offspring_len - 1), random.randint(0, offspring_len - 1)
                            while offspring[pos2] == offspring[pos1]:
                                pos1, pos2 = random.randint(0, offspring_len - 1), random.randint(0, offspring_len - 1)
                            tmp = offspring[pos1]
                            offspring[pos1] = offspring[pos2]
                            offspring[pos2] = tmp
                        elif status:  # insert
                            pos1, pos2 = random.randint(0, offspring_len - 1), random.randint(0, offspring_len - 1)
                            status = True
                            while status:
                                if pos1 == pos2:
                                    pos1, pos2 = random.randint(0, offspring_len - 1), random.randint(0,
                                                                                                      offspring_len - 1)
                                else:
                                    if pos1 < pos2:
                                        sub1, sub2 = pos1, pos2
                                    else:
                                        sub1, sub2 = pos2, pos1
                                    begin = offspring[sub1]
                                    for i in range(sub1, sub2 + 1):
                                        if offspring[i] != begin:
                                            status = False
                                            break
                                if status:
                                    pos1, pos2 = random.randint(0, offspring_len - 1), random.randint(0,
                                                                                                      offspring_len - 1)
                            tmp = offspring[pos1]
                            del offspring[pos1]
                            if pos2 == offspring_len - 1:
                                offspring.append(tmp)
                            else:
                                offspring.insert(pos2, tmp)
                        else:
                            mutation_type = random.randint(1, 2)
                    else:
                        mutation_type = random.randint(1, 2)
                if mutation_type == 1:
                    if offspring_len > 1:  # alter
                        pos = random.randint(0, offspring_len - 1)
                        tmp = offspring[pos]
                        while offspring[pos] == tmp:
                            offspring[pos] = random.randint(0, 2)
                    else:
                        mutation_type = 2
                if mutation_type == 2:  # add
                    pos = random.randint(0, offspring_len)
                    block_type = random.randint(0, 2)
                    if pos == offspring_len:
                        offspring.append(block_type)
                    else:
                        offspring.insert(pos, block_type)

                offspring_str = ''
                for num in offspring:
                    offspring_str += '%d' % num

                if mutation_type == 0:
                    self.log.info('\tOffspring {} through adjust-process, offspring {} with lr={}, wd={}, dropout={}, hidden_dim={} is generated.'.format(offspring_before_str, offspring_str, self.info[index][0], self.info[index][1], self.info[index][2], self.info[index][3]))
                elif mutation_type == 1:
                    self.log.info('\tOffspring {} through alter-process, offspring {} with lr={}, wd={}, dropout={}, hidden_dim={} is generated.'.format(offspring_before_str, offspring_str, self.info[index][0], self.info[index][1], self.info[index][2], self.info[index][3]))
                elif mutation_type == 2:
                    self.log.info('\tOffspring {} through add-process, offspring {} with lr={}, wd={}, dropout={}, hidden_dim={} is generated.'.format(offspring_before_str, offspring_str, self.info[index][0], self.info[index][1], self.info[index][2], self.info[index][3]))
                else:
                    raise Exception('error')
        return self.offsprings, self.info

