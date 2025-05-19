import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import *
from utils.metric import pixel_accuracy, intersection_over_union
import numpy as np


def l1_distance(feature1, feature2):
    abs_diff = torch.abs(feature1 - feature2)
    l1_score = torch.mean(abs_diff, dim=-1)
    return l1_score


def l2_distance(feature1, feature2):
    squared_diff = (feature1 - feature2)**2
    l2_score = torch.mean(squared_diff, dim=-1)
    return squared_diff


def cosine_distance(feature1, feature2):
    cosine_sim = F.cosine_similarity(feature1, feature2, dim=-1)
    cosine_dist = 1.0 - cosine_sim
    return cosine_dist


def kl_div(source, target, reduction='batchmean'):
    loss = F.kl_div(F.log_softmax(source, 1), target, reduction=reduction)
    return loss

# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss(reduction='mean')
    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss


def test(models, method, dataloaders, mode='test'):
    device = torch.device(CUDA if torch.cuda.is_available() else 'cpu')

    assert mode in ['val', 'test']
    models['backbone'].eval()
    if method in ['TiDAL', 'lloss']:
        models['module'].eval()

    with torch.no_grad():
        preds = torch.tensor([]).to(device)
        labels = torch.tensor([]).to(device)
        for data in dataloaders[mode]:
            input1 = data[0].to(device)
            input2 = data[1].to(device)
            label = data[2].to(device)

            if method == 'DAAL':
                all_votes = []
                pred1, _, _, _ = models['backbone'](input1)
                all_votes.append(pred1)
                for _ in range(VOTE):
                    _, pred2, _, _ = models['backbone'](input2)
                    all_votes.append(pred2)
                votes_tensor = torch.stack(all_votes, dim=0)
                final_pred, _ = torch.mode(votes_tensor, dim=0)

                preds = torch.cat((preds, final_pred), 0)
                labels = torch.cat((labels, label), 0)
            else:
                pred1, _, _, _ = models['backbone'](input1)

                preds = torch.cat((preds, pred1), 0)
                labels = torch.cat((labels, label), 0)

        acc = pixel_accuracy(preds, labels)
        miou = intersection_over_union(preds, labels, num_classes=2)

    return acc, miou


def train_epoch(models, method, criterion, optimizers, dataloaders, epoch):
    device = torch.device(CUDA if torch.cuda.is_available() else 'cpu')
    models['backbone'].train()
    if method in ['TiDAL', 'lloss']:
        models['module'].train()

    for data in dataloaders['train']:
        inputs1 = data[0].to(device)
        inputs2 = data[1].to(device)
        labels = data[2].to(device)
        
        optimizers['backbone'].zero_grad()
        if method in ['TiDAL', 'lloss']:
            optimizers['module'].zero_grad()

        pred1, _, embed1, features1 = models['backbone'](inputs1)
        _, pred2, embed2, _ = models['backbone'](inputs2)

        target_loss1 = criterion['CE'](pred1, labels)
        target_loss2 = criterion['CE'](pred2, labels)

        if method == 'DAAL':
            target_loss1 = torch.mean(torch.sum(target_loss1, dim=(-1,-2)))
            target_loss2 = torch.mean(torch.sum(target_loss2, dim=(-1,-2)))
            assert DIS in ['L1', 'L2', 'Cos']
            embed1 = torch.mean(embed1, dim=(-1, -2))
            embed2 = torch.mean(embed2, dim=(-1, -2))
            if DIS == 'L1':
                m_module_loss = l1_distance(embed1, embed2)
            elif DIS == 'L2':
                m_module_loss = l2_distance(embed1, embed2)
            elif DIS == 'Cos':
                m_module_loss = cosine_distance(embed1, embed2)
            m_module_loss = torch.mean(m_module_loss)
            m_backbone_loss = target_loss1 + target_loss2
            loss = m_backbone_loss - LAMBDA * m_module_loss

        elif method == 'TiDAL':
            index = data[3].detach().numpy().tolist()
            moving_prob = data[4].to(device)
            prob1 = F.softmax(pred1, dim=1)
            moving_prob = (moving_prob * epoch + prob1 * 1) / (epoch + 1)
            dataloaders['train'].dataset.moving_prob[index, :] = moving_prob.cpu().detach().numpy()
            cumulative_logit = models['module'](features1)
            m_module_loss = criterion['KL_Div'](F.log_softmax(cumulative_logit, dim=1), moving_prob.detach())
            m_backbone_loss = torch.mean(torch.sum(target_loss1, dim=(-1,-2)))
            loss = m_backbone_loss + TDWEIGHT * m_module_loss

        elif method == 'lloss':
            pred_loss = models['module'](features1)
            pred_loss = pred_loss.view(pred_loss.size(0))
            target_loss_reduc = torch.sum(target_loss1, dim=(-1,-2)) 
            m_module_loss = LossPredLoss(pred_loss, target_loss_reduc, margin=MARGIN)
            m_backbone_loss = torch.mean(torch.sum(target_loss1, dim=(-1,-2)))
            loss = m_backbone_loss + LLWEIGHT * m_module_loss

        else:
            m_backbone_loss = torch.mean(torch.sum(target_loss1, dim=(-1,-2)))
            loss = m_backbone_loss

        loss.backward()
        optimizers['backbone'].step()
        if method in ['TiDAL', 'lloss']:
            optimizers['module'].step()
            
    return loss.item()


def train(models, method, criterion, optimizers, dataloaders, num_epochs):
    print('>> Train a Model.')

    minloss = np.inf
    count = 0
    for epoch in tqdm(range(num_epochs)):
        loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch)

        if loss < minloss:
            minloss = loss
            count = 0
        else:
            count += 1

        if count >= 10:
            break

        # print(loss, count)

    print('>> Finished.')
