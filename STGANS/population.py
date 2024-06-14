import copy
from model import EvoGCN
import random


class Individual(object):
    def __init__(self, args, num_node_features, num_classes):
        self.args = args
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.fitness = 99999
        self.hidden_dim = 0
        self.dropout = 0.0
        self.lr = 0.0
        self.weight_decay = 0.0

    def initialize(self, genotype=None, lr=None, weight_decay=None, dropout=None, hidden_dim=None):
        if genotype is not None:
            self.genotype = genotype
            self.dropout = dropout
            self.lr = lr
            self.weight_decay = weight_decay
            self.hidden_dim = hidden_dim
        else:
            self.len = random.randint(1, self.args.max_len)
            self.genotype = []
            for _ in range(self.len):
                self.genotype.append(random.randint(0, 2))
            while self.dropout <= 0.0 or self.dropout >= 1.0:
                self.dropout = round(random.random(), 1)
            while self.lr == 0.0:
                choose = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
                r = random.randint(0, 9)
                self.lr = choose[r]
            while self.weight_decay == 0.0:
                choose = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
                r = random.randint(0, 5)
                self.weight_decay = choose[r]
            choose = [16, 32, 64, 128, 256, 512]
            self.hidden_dim = choose[random.randint(0, 5)]
        self.get_record(self.genotype)

        self.phenotype = EvoGCN(self.args, self.record, self.num_node_features, self.num_classes, self.dropout, self.hidden_dim)  # 网络

    def get_record(self, genotype):
        self.record = []
        elem = genotype[0]
        num = 1
        subs = 1
        while subs < len(genotype):
            if genotype[subs] == elem:
                num += 1
                if subs == len(genotype) - 1:
                    self.record.append([elem, num])
                subs += 1
            else:
                self.record.append([elem, num])
                elem = genotype[subs]
                if subs == len(genotype) - 1:
                    self.record.append([elem, 1])
                    subs += 1
                else:
                    subs += 1
                    num = 1


class Population(object):
    def __init__(self, args, num_node_features, num_classes, gen_no):
        self.args = args
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.gen_no = gen_no
        self.pop_size = self.args.pop_size
        self.individuals = []

    def initialize(self, genotype_records, ind_records):
        while len(self.individuals) < self.pop_size:
            indi = Individual(self.args, self.num_node_features, self.num_classes)
            indi.initialize()
            if str(indi.genotype) not in genotype_records:
                self.individuals.append(indi)
                genotype_records.append(str(indi.genotype))
                ind_record = str(indi.genotype) + str(indi.lr) + str(indi.weight_decay) + str(indi.dropout) + str(indi.hidden_dim)
                ind_records.append(ind_record)
        return genotype_records, ind_records

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            self.individuals.append(indi)
