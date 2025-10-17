import argparse
import torch
import os
from population import Population
from evaluate import FitnessEvaluate
from crossover_and_mutation import CrossoverAndMutation
from selection import Selection
from utils import Log
import copy
import numpy as np

from torch_geometric.nn import GATConv

from dataset_loader import DataLoader

parser = argparse.ArgumentParser(description='Implementation of Evolutionary Graph Convolution Networks')
parser.add_argument('--dataset', type=str, default='Photo', help='Cora or CiteSeer or PubMed or PPI ...')
parser.add_argument('--task_type', type=str, default='full', help='semi or full')
parser.add_argument('--data_path', type=str, default='data')
parser.add_argument('--pop_size', type=int, default=30)
parser.add_argument('--max_gen', type=int, default=20)
parser.add_argument('--max_len', type=int, default=10)
parser.add_argument('--res_block_len', type=int, default=2)
parser.add_argument('--dense_block_len', type=int, default=2)
parser.add_argument('--incep_block_len', type=int, default=2)
parser.add_argument('--pc', type=float, default=0.9, help='Crossover probability')
parser.add_argument('--pm', type=float, default=0.2, help='Mutation probability')
parser.add_argument('--total_epochs', type=int, default=400, help='number of total epochs to run')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--show', type=bool, default=True, help='show the training process')
parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
args = parser.parse_args()

dataset = DataLoader(args.dataset)
data = dataset[0]

class EvoGCN(object):
    def __init__(self, arg):
        self.gen_no = 0
        self.args = arg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fitness_records = {}
        self.indi_fitness_records = []
        self.genotype_records = []
        self.ind_records = []
        self.best_val2test = 0
        self.best_fitness = 99999
        self.best_genotype = ''
        self.best_lr = 0.0
        self.best_wd = 0.0
        self.best_dropout = 0.0
        self.best_hidden_dim = 0

    def initialize_population(self):
        self.dataset = DataLoader(args.dataset)
        self.data = self.dataset[0]
        self.num_node_features = dataset.num_features
        self.num_classes = dataset.num_classes
        pops = Population(self.args, self.num_node_features, self.num_classes, 0)
        self.genotype_records, self.ind_records = pops.initialize(self.genotype_records, self.ind_records)
        self.pops = pops

    def fitness_evaluate(self):
        indi_no = 0
        indi_fitness_record = []
        for indi in self.pops.individuals:
            fitness = FitnessEvaluate(indi, self.device, Log, self.data, self.dataset, args)
            indi.fitness, self.best_fitness, self.best_genotype, self.fitness_records, self.best_lr, self.best_wd, self.best_dropout, self.best_val2test, self.best_hidden_dim = fitness.evaluate(
                self.args, self.gen_no, indi_no, self.fitness_records, self.best_fitness, self.best_genotype,
                self.best_lr,
                self.best_wd, self.best_dropout, self.best_val2test, self.best_hidden_dim)
            indi_no += 1
            indi_fitness_record.append(indi.fitness)
        self.indi_fitness_records.append(indi_fitness_record)

    def crossover_and_mutation(self):
        cm = CrossoverAndMutation(self.pops.individuals, self.args, self.num_node_features, self.num_classes,
                                  self.gen_no, Log)
        offsprings, self.genotype_records, self.ind_records = cm.process(self.genotype_records, self.ind_records)
        self.parent_pops = copy.deepcopy(self.pops)
        self.pops.individuals = copy.deepcopy(offsprings)

    def environment_selection(self):
        selection = Selection(self.parent_pops.individuals, self.pops.individuals)
        next_individuals = selection.do_selection(self.args)
        next_gen_pops = Population(self.args, self.num_node_features, self.num_classes, self.gen_no + 1)
        next_gen_pops.create_from_offspring(next_individuals)
        self.pops = next_gen_pops

    def run(self):
        Log.info('Initialize population ...')
        self.initialize_population()
        Log.info('Evolve {:02d}-gen. Begin to evaluate the fitness'.format(self.gen_no))
        self.fitness_evaluate()
        Log.info('The best fitness is {:.3f}, the corresponding acc_test={}, genotype={}, hidden_dim={}, dropout={}, lr={}, weight decay={}.'.format(
                self.best_fitness, self.best_val2test, self.best_genotype, self.best_hidden_dim, self.best_dropout, self.best_lr, self.best_wd))
        for cur_gen in range(1, self.args.max_gen):
            self.gen_no = cur_gen
            Log.info('Evolve {:02d}-gen. Begin to crossover and mutation'.format(self.gen_no))
            self.crossover_and_mutation()
            Log.info('Evolve {:02d}-gen. Begin to evaluate the fitness'.format(cur_gen))
            self.fitness_evaluate()
            Log.info('Evolve {:02d}-gen. Begin to the environment selection'.format(self.gen_no))
            self.environment_selection()
            Log.info('The best fitness is {:.3f}, the corresponding acc_test={}, genotype={}, hidden_dim={}, dropout={}, lr={}, weight decay={}.'.format(
                    self.best_fitness, self.best_val2test, self.best_genotype, self.best_hidden_dim, self.best_dropout, self.best_lr, self.best_wd))

        Log.info('{} individuals in total. Finish the evaluation.'.format(len(self.fitness_records.keys())))


def main():
    if os.path.exists('log.txt'):
        os.remove('log.txt')
    seed = np.random.randint(0, 10000)
    args.seed = seed
    Log.info('args = {}'.format(args))
    evoGCN = EvoGCN(args)
    evoGCN.run()


if __name__ == '__main__':
    main()
