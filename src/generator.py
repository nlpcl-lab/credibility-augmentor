import os
import argparse
import random
from collections import Counter, defaultdict

import jsonlines
import torch
import torch.nn as nn
from tqdm import tqdm, trange

import things
from dataset import PfGDataset, PfGFeatureDataset, pfg_features
from models import BertForPersuasiveConnection
from train import pfg_feature_collator

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--bert', type=str, default='bert-base-uncased')
    parser.add_argument('--data', type=str, default='../data/test.jsonl')
    parser.add_argument('--mf', type=str, default='../models')
    parser.add_argument('--out', type=str, default='../generated')

    args = parser.parse_args()
    return args


STRATEGIES = [None, '[EOP]'] + things.STRATEGIES
COMMON_CONS = [None, '[EOP]'] + things.COMMON_CONS


def generate(args):
    model = BertForPersuasiveConnection.from_pretrained(args.mf)
    model.to(DEVICE)
    model.eval()

    dataset = PfGDataset(args.data)
    dataset_by_strategies = defaultdict(list)

    for data in dataset:
        dataset_by_strategies[data['strategy']].append(data)
    dataset_by_strategies['[EOP]'].append({
        'id': '?-?',
        'er_sentence': None,
        'ee_sentence': None,
        'strategy': '[EOP]',
        'sentiment': None,
        'connective': None,
        'turn': None,
        'next_strategy': None,
        'next_connective': None
    })

    feature_dataset = PfGFeatureDataset(PfGDataset(
        args.data), bert_type=args.bert, lazy=False, return_sample=False)
    data_loader = torch.utils.data.DataLoader(
        feature_dataset, batch_size=1, num_workers=1, collate_fn=pfg_feature_collator)

    writer = jsonlines.open(os.path.join(args.out, 'enhanced.jsonl'), mode='w')

    with torch.no_grad():
        for idx, feature_data in enumerate(tqdm(iter(data_loader))):
            data = dataset[idx]

            (input_ids,
             attention_mask,
             token_type_ids,
             next_strategy_idx,
             next_connective_idx,
             strategy_changed) = feature_data

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

            next_data = random.choice(dataset_by_strategies[STRATEGIES[strt]])

            writer.write([data, next_data])

    writer = jsonlines.open(os.path.join(args.out, 'baseline.jsonl'), mode='w')

    for data in dataset:
        next_data = random.choice(dataset_by_strategies['credibility-appeal'])
        writer.write([data, next_data])


if __name__ == '__main__':
    args = _get_args()
    generate(args)
