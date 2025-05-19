# Python
import os
import random
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from models.unet import U_Net
from models.query_models import LossNet, TDNet
from utils.train_test import train, test
from data.load_dataset import load_dataset
from methods.selection_methods import query_samples
from config import *


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="InES", help="InES, CVC, InES2CVC")
parser.add_argument("-m", "--method_type", type=str, default="DAAL", help="Random, Entropy, CoreSet, lloss, TiDAL, DAAL")
parser.add_argument("-c", "--cycles", type=int, default=CYCLES, help="Number of active learning cycles")
parser.add_argument("-i", "--initial_size", type=int, default=INIT, help="Number of initial size")
parser.add_argument("-a", "--add_num", type=int, default=ADD, help="Number of add size")
parser.add_argument("-s", "--subset", type=int, default=SUBSET, help="The size of subset.")
parser.add_argument("-w", "--num_workers", type=str, default=0, help="The number of workers.")

args = parser.parse_args()

# Main
if __name__ == '__main__':

    method = args.method_type
    methods = ['Random', 'Entropy', 'CoreSet', 'lloss', 'TiDAL', 'DAAL']
    dataset = args.dataset
    datasets = ['InES', 'CVC', 'InES2CVC']
    assert method in methods, 'No method %s! Try options %s' % (method, methods)
    assert dataset in datasets, 'No dataset %s! Try options %s' % (dataset, datasets)

    os.makedirs('results', exist_ok=True)
    txt_name = f'results/results_{dataset}_{method}.txt'
    results = open(txt_name, 'w')
    results.write('method cur_trial all_trial cur_cycle all_cycle num_labels acc iou')
    results.write('\n')

    print(txt_name)
    print("Dataset: %s" % dataset)
    print("Method type: %s" % method)

    for trial in range(TRIALS):

        # Load training and testing dataset
        data_train, data_unlabeled, data_test, data_valid, adden, no_train = load_dataset(args)
        print('The entire datasize is {}'.format(len(data_train)))
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))

        labeled_set = indices[:args.add_num]
        unlabeled_set = [x for x in indices if x not in labeled_set]

        train_loader = DataLoader(data_train, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True, num_workers=args.num_workers)

        test_loader = DataLoader(data_test, batch_size=BATCH,
                                 pin_memory=True, num_workers=args.num_workers)
        
        if MODE == "test":
            test_loader = DataLoader(data_test, batch_size=BATCH,
                                     pin_memory=True, num_workers=args.num_workers)
            dataloaders = {'train': train_loader, 'test': test_loader}
        elif MODE == "val":
            valid_loader = DataLoader(data_valid, batch_size=BATCH,
                                    pin_memory=True, num_workers=args.num_workers)
            dataloaders = {'train': train_loader, 'val': valid_loader}
        else:
            raise ValueError('MODE is must set to test/val !')

        # Model - create new instance for every trial so that it resets
        device = torch.device(CUDA if torch.cuda.is_available() else 'cpu')

        unet = U_Net(in_channels=3, out_channel=2, init_features=32).to(device)

        if method == 'lloss':
            pred_module = LossNet().to(device) # prediction module for lloss
            models = {'backbone': unet, 'module': pred_module}
        elif method == 'TiDAL':
            pred_module = TDNet().to(device) # prediction module for TiDAL
            models = {'backbone': unet, 'module': pred_module}
        else:
            models = {'backbone': unet}

        # Loss and criterion
        criterion = {}
        criterion['CE'] = nn.CrossEntropyLoss(reduction='none')
        criterion['KL_Div'] = nn.KLDivLoss(reduction='batchmean') # for TiDAL

        for key, val in models.items():
            models[key] = models[key].to(device)

        for cycle in range(CYCLES):
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:args.subset]

            optim_backbone = optim.Adam(models['backbone'].parameters(), lr=LR, weight_decay=WDECAY)
            if method in ['TiDAL', 'lloss']:
                optim_module = optim.Adam(models['module'].parameters(), lr=LR, weight_decay=WDECAY)
                optimizers = {'backbone': optim_backbone, 'module': optim_module}
            else:
                optimizers = {'backbone': optim_backbone}

            # Training and testing
            train(models, method, criterion, optimizers, dataloaders, EPOCH)

            # Save param
            torch.save(models['backbone'].state_dict(), 'weights/'+method+'/Trial'+str(trial + 1) + '_Cycle' + str(cycle + 1) + '.pth')
            
            acc, iou = test(models, method, dataloaders, mode=MODE)
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: \nTest m_acc: {} || m_iou: {} '.format(
                trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc, iou))
            
            np.array([method, trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc, iou]).tofile(results, sep=" ")
            results.write("\n")

            # Reached final training cycle
            if cycle == (CYCLES - 1):
                print("Finished.")
                break

            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-args.add_num:].numpy())
            listd = list(torch.tensor(subset)[arg][:-args.add_num].numpy())
            unlabeled_set = listd + unlabeled_set[args.subset:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True, num_workers=args.num_workers)

    results.close()
