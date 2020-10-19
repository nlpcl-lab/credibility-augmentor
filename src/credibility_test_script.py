import os
import time
import random
import jsonlines

import torch
import torch.nn as nn
from tqdm import tqdm, trange

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script

from dataset import PfGDataset, PfGFeatureDataset
from models import BertForPersuasiveConnection
from train import pfg_feature_collator

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_success_rate(dataset, data_loader, model):
    with torch.no_grad():
        total = 0
        success = 0

        for data in tqdm(iter(data_loader)):
            (input_ids,
             attention_mask,
             token_type_ids,
             _,
             _,
             strategy_changing_decision) = data

            out = model(
                input_ids.to(DEVICE),
                attention_mask.to(DEVICE),
                token_type_ids.to(DEVICE)
            )
            _, _, decision = out

            softmax = nn.Softmax(dim=1)
            decision = torch.argmax(softmax(decision), dim=1).cpu().item()

            if decision == strategy_changing_decision.item():
                success += 1
            total += 1

    return {
        'num_of_dialogues': len(dataset),
        'success_rate': success / total
    }


def get_strt_acc(documents):
    success = 0
    for document in documents:
        previous = document[0]
        after = document[1]

        if previous['next_strategy'] == after['strategy']:
            success += 1

    return success / len(documents)


def get_enhance_rate(generated_path):
    # data_dict = {x['id']: x for x in dataset}

    # for idx, feature_data in enumerate(tqdm(iter(data_loader))):
    #     data = dataset[idx]
    #     data_dict[data['id']] = [data, feature_data]

    reader = jsonlines.open(os.path.join(generated_path, 'enhanced.jsonl'))
    documents = [x for x in reader]
    enhanced_acc = get_strt_acc(documents)

    reader = jsonlines.open(os.path.join(generated_path, 'baseline.jsonl'))
    documents = [x for x in reader]
    baseline_acc = get_strt_acc(documents)

    return {
        'num_of_documents': len(documents),
        'enhance_rate': enhanced_acc / baseline_acc - 1.0,
        'baseline_acc': baseline_acc,
        'enhanced_acc': enhanced_acc
    }


@register_script('credibility_test')
class CredibilityTest(ParlaiScript):
    @classmethod
    def setup_args(cls):
        parser = ParlaiParser(True, True)
        parser.add_argument(
            '--generated_path', type=str, help='path of generated documents'
        )
        parser.add_argument('--test_data_path', type=str,
                            help='test data path')
        parser.add_argument(
            '--mf', type=str, help='model file', default='../models')
        return parser

    def run(self):
        start = time.time()

        model = BertForPersuasiveConnection.from_pretrained(self.opt['mf'])
        model.to(DEVICE)
        model.eval()

        dataset = PfGDataset(self.opt['test_data_path'])
        feature_dataset = PfGFeatureDataset(
            dataset, bert_type='bert-base-uncased', lazy=False, return_sample=False)
        data_loader = torch.utils.data.DataLoader(feature_dataset,
                                                  batch_size=1,
                                                  num_workers=1,
                                                  collate_fn=pfg_feature_collator)

        reader = jsonlines.open(self.opt['test_data_path'])
        dialogues = [x for x in reader]
        print('\nComparing %d Documents...\n' % len(dialogues))

        success_stats = get_success_rate(
            dataset, data_loader, model)
        enhance_stats = get_enhance_rate(
            self.opt['generated_path'])

        processing_time = time.time() - start

        print('\n## Distribution Information ##')
        print('Number of dialogues: %d' % len(dataset))

        print('\n## Main Results ##')
        print('\nSuccess\tRate: %f' % success_stats['success_rate'])
        print('Increase Rate: %f' % enhance_stats['enhance_rate'])

        print('\nTotal processing time: %f sec' % processing_time)
        return success_stats, enhance_stats, processing_time


if __name__ == '__main__':
    CredibilityTest.main()
