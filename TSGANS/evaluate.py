import torch
from torch.nn import functional
import numpy as np
from utils import random_planetoid_splits


class FitnessEvaluate(object):
    def __init__(self, individual, device, Log, data, dataset, args):
        self.indi = individual
        self.device = device
        self.log = Log
        self.dataset = dataset

        percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
        val_lb = int(round(args.val_rate * len(data.y)))
        permute_masks = random_planetoid_splits
        self.data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

        self.labels, self.idx_train, self.idx_val, self.idx_test = self.data.y, self.data.train_mask, self.data.val_mask, self.data.test_mask

        #self.labels, self.idx_train, self.idx_val, self.idx_test = self.sampler.get_label_and_idxes()

    def evaluate(self, args, gen_no, indi_no, fitness_records, best_fitness, best_genotype, best_lr, best_wd,
                 best_dropout, best_val2test, best_hidden_dim):
        genotype_str = ''
        for num in self.indi.genotype:
            genotype_str += '%d' % num
        model_paras_str = genotype_str + str(self.indi.lr) + str(self.indi.weight_decay) + str(self.indi.dropout) + str(self.indi.hidden_dim)
        self.log.info('\tIndividual gen{}--{}, genotype={}, hidden_dim={}, dropout={}, lr={}, weight decay={} begin to evaluate.'.format(
            gen_no, indi_no, genotype_str, self.indi.hidden_dim, self.indi.dropout, self.indi.lr, self.indi.weight_decay))
        if model_paras_str in fitness_records.keys():
            self.log.info('\t\tIndividual {} has exist, skip it.'.format(model_paras_str))
            return fitness_records[model_paras_str], best_fitness, best_genotype, fitness_records, best_lr, best_wd, best_dropout, best_val2test, best_hidden_dim
        try:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            model = self.indi.phenotype.to(self.device)
            labels = self.labels.to(self.device)
            idx_train = self.idx_train.to(self.device)
            idx_val = self.idx_val.to(self.device)
            idx_test = self.idx_test.to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.indi.lr, weight_decay=self.indi.weight_decay)
            fitness = 99999
            best_epoch = 0
            status = False
            _best_fitness = best_fitness
            save_path = 'data/' + args.dataset.lower() + '/' + args.dataset.lower() + '_model.pt'

            for epoch in range(args.total_epochs):
                train_adj, train_fea = self.data.edge_index, self.data.x
                train_adj, train_fea = train_adj.to(self.device), train_fea.to(self.device)
                val_adj, val_fea = self.data.edge_index, self.data.x
                val_adj, val_fea = val_adj.to(self.device), val_fea.to(self.device)

                model.train()
                optimizer.zero_grad()
                out = model(train_fea, train_adj)
                loss_train = functional.nll_loss(out[idx_train], labels[idx_train])
                loss_train.backward()
                optimizer.step()
                loss_train = loss_train.item()

                model.eval()
                output = model(val_fea, val_adj)
                loss_val = functional.nll_loss(output[idx_val], labels[idx_val])
                loss_val = loss_val.item()

                if args.show:
                    if (epoch + 1) % 100 == 0 or epoch == 0:
                        self.log.info('\t\tEpoch: {:03d}, Train loss: {:.3f}, Val loss: {:.3f}'.format(epoch + 1, loss_train, loss_val))
                if loss_val < fitness:
                    fitness = loss_val
                    best_epoch = epoch + 1
                    if fitness < best_fitness:

                        status = True
                        model.eval()
                        test_adj, test_fea = self.data.edge_index, self.data.x
                        test_adj, test_fea = test_adj.to(self.device), test_fea.to(self.device)
                        output = model(test_fea, test_adj)
                        predict = output[idx_test].max(1)[1].type_as(labels[idx_test])
                        correct = predict.eq(labels[idx_test]).double()
                        acc_test = correct.sum() / len(labels[idx_test])
                        acc_test = acc_test.item()

                        best_fitness = fitness
                        best_val2test = acc_test
                        best_genotype = genotype_str
                        best_lr = self.indi.lr
                        best_wd = self.indi.weight_decay
                        best_dropout = self.indi.dropout
                        best_hidden_dim = self.indi.hidden_dim
                        torch.save(model.state_dict(), save_path)
            if status:
                self.log.info('\t\tBest val loss improves from {:.3f} to {:.3f}, corresponding acc test {:.3f}.'.format(
                    _best_fitness, best_fitness, best_val2test))
            self.log.info('\t\tBVal loss: {:.3f}, best epoch: {}'.format(fitness, best_epoch))
            fitness_records[model_paras_str] = fitness
            return fitness, best_fitness, best_genotype, fitness_records, best_lr, best_wd, best_dropout, best_val2test, best_hidden_dim
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                self.log.info('\t\tIndividual {} out of memory, skip it.'.format(genotype_str))
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                return 99999, best_fitness, best_genotype, fitness_records, best_lr, best_wd, best_dropout, best_val2test, best_hidden_dim
            else:
                raise exception
