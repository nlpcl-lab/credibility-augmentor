import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm, trange

from dataset import PfGDataset, PfGFeatureDataset
from models import BertForPersuasiveConnection

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--bert', type=str, default='bert-base-uncased')
    # parser.add_argument('--data', type=str, default='../pfg.jsonl')
    parser.add_argument('--mf', type=str, default='../models')
    parser.add_argument('--train', type=str, default='../data/train.jsonl')
    parser.add_argument('--dev', type=str, default='../data/dev.jsonl')
    parser.add_argument('--test', type=str, default='../data/test.jsonl')

    parser.add_argument('--eval_only', action='store_true', default=False)

    args = parser.parse_args()
    return args


def pfg_feature_collator(samples):
    input_ids, token_type_ids, attention_mask, \
        next_strategy, next_connective, strategy_changed = zip(*samples)

    input_ids = pad_sequence([torch.Tensor(input_id).to(torch.long)
                              for input_id in input_ids],
                             padding_value=0, batch_first=True)
    token_type_ids = pad_sequence([torch.Tensor(input_id).to(torch.long)
                                   for input_id in token_type_ids],
                                  padding_value=0, batch_first=True)
    attention_mask = pad_sequence([torch.Tensor(input_id).to(torch.long)
                                   for input_id in attention_mask],
                                  padding_value=0, batch_first=True)

    next_strategy = torch.Tensor(next_strategy).to(torch.long)
    next_connective = torch.Tensor(next_connective).to(torch.long)
    strategy_changed = torch.Tensor(strategy_changed).to(torch.long)

    return (input_ids,
            attention_mask,
            token_type_ids,
            next_strategy,
            next_connective,
            strategy_changed)


def train(args):
    model = BertForPersuasiveConnection.from_pretrained(args.bert)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.to(DEVICE)

    train_dataset = PfGFeatureDataset(PfGDataset(
        args.train), bert_type=args.bert, lazy=False, return_sample=False)
    dev_dataset = PfGFeatureDataset(PfGDataset(
        args.dev), bert_type=args.bert, lazy=False, return_sample=False)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.bs,
                                                    num_workers=4,
                                                    collate_fn=pfg_feature_collator)
    dev_data_loader = torch.utils.data.DataLoader(dev_dataset,
                                                  batch_size=1,
                                                  num_workers=1,
                                                  collate_fn=pfg_feature_collator)

    summary = SummaryWriter()
    index = -1
    for _ in trange(args.epoch):
        strt_loss_per_epoch = 0.
        conn_loss_per_epoch = 0.
        chng_loss_per_epoch = 0.
        loss_per_epoch = 0.

        highest_accuracy = 0.

        for data in tqdm(iter(train_data_loader)):
            index += 1
            model.train()

            (input_ids,
             attention_mask,
             token_type_ids,
             next_strategy_idx,
             next_connective_idx,
             strategy_changed) = data

            optimizer.zero_grad()
            out = model(
                input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
                token_type_ids.to(DEVICE)
            )
            pred_strt, pred_conn, pred_chng = out

            loss_fn = nn.CrossEntropyLoss()
            strt_loss = loss_fn(pred_strt, next_strategy_idx.to(DEVICE))
            conn_loss = loss_fn(pred_conn, next_connective_idx.to(DEVICE))
            chng_loss = loss_fn(pred_chng, strategy_changed.to(DEVICE))
            loss = strt_loss + conn_loss + chng_loss

            loss.backward()
            optimizer.step()

            if index % 200 == 0:
                summary.add_scalar('loss/strt', strt_loss, index)
                summary.add_scalar('loss/conn', conn_loss, index)
                summary.add_scalar('loss/chng', chng_loss, index)
                summary.add_scalar('loss/total', loss, index)

                model.eval()

                with torch.no_grad():
                    total = 0
                    correct_strts = 0
                    correct_conns = 0
                    correct_chngs = 0

                    for dev_data in iter(dev_data_loader):
                        (dev_input_ids,
                         dev_attention_mask,
                         dev_token_type_ids,
                         dev_next_strategy_idx,
                         dev_next_connective_idx,
                         dev_strategy_changed) = dev_data

                        dev_out = model(
                            dev_input_ids.to(DEVICE),
                            dev_attention_mask.to(DEVICE),
                            dev_token_type_ids.to(DEVICE)
                        )
                        dev_pred_strt, dev_pred_conn, dev_pred_chng = dev_out

                        softmax = nn.Softmax(dim=1)

                        strt = torch.argmax(
                            softmax(dev_pred_strt), dim=1).cpu().item()
                        conn = torch.argmax(
                            softmax(dev_pred_conn), dim=1).cpu().item()
                        chng = torch.argmax(
                            softmax(dev_pred_chng), dim=1).cpu().item()

                        if strt == dev_next_strategy_idx.item():
                            correct_strts += 1
                        if conn == dev_next_connective_idx.item():
                            correct_conns += 1
                        if chng == dev_strategy_changed.item():
                            correct_chngs += 1
                        total += 1

                    summary.add_scalar(
                        'acc/strt', correct_strts / total, index)
                    summary.add_scalar(
                        'acc/conn', correct_conns / total, index)
                    summary.add_scalar(
                        'acc/chng', correct_chngs / total, index)

                    accuracy_sum = (correct_strts + correct_conns +
                                    correct_chngs) / total

                    if accuracy_sum > highest_accuracy and index / 100 > 12:
                        highest_accuracy = accuracy_sum
                        model.save_pretrained(args.mf)

    # model.save_pretrained(args.mf)
    summary.close()


def eval(args):
    model = BertForPersuasiveConnection.from_pretrained(args.mf)
    model.to(DEVICE)

    model.eval()

    test_dataset = PfGFeatureDataset(PfGDataset(
        args.test), bert_type=args.bert, lazy=False, return_sample=False)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=1,
                                                   num_workers=1,
                                                   collate_fn=pfg_feature_collator)

    with torch.no_grad():
        total = 0
        correct_strts = 0
        correct_conns = 0
        correct_chngs = 0

        for data in tqdm(iter(test_data_loader)):
            (input_ids,
             attention_mask,
             token_type_ids,
             next_strategy_idx,
             next_connective_idx,
             strategy_changed) = data

            out = model(
                input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
                token_type_ids.to(DEVICE)
            )
            pred_strt, pred_conn, pred_chng = out

            softmax = nn.Softmax(dim=1)

            strt = torch.argmax(softmax(pred_strt), dim=1).cpu().item()
            conn = torch.argmax(softmax(pred_conn), dim=1).cpu().item()
            chng = torch.argmax(softmax(pred_chng), dim=1).cpu().item()

            if strt == next_strategy_idx.item():
                correct_strts += 1
            if conn == next_connective_idx.item():
                correct_conns += 1
            if chng == strategy_changed.item():
                correct_chngs += 1
            total += 1

        print('\n=== TEST ACCURACY ===')
        print('STRATEGY:\t%f' % (correct_strts / total))
        print('CONNECTIVE:\t%f' % (correct_conns / total))
        print('CHANGABLE:\t%f' % (correct_chngs / total))


if __name__ == '__main__':
    args = _get_args()
    if not args.eval_only:
        train(args)
    eval(args)
