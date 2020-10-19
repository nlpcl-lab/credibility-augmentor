import jsonlines
from tqdm import trange
from transformers import BertTokenizerFast

import things


def filter_label(labels):
    for label, _ in labels:
        if label in things.STRATEGIES:
            return label
    return None


class PfGData(object):
    def __init__(self, er_data, ee_data):
        self.er_sentence = er_data['sentence']
        self.strategy = filter_label(er_data['label'])
        self.connective = er_data['connective']
        self.id = None

        if ee_data:
            self.ee_sentence = ee_data['sentence']
            self.sentiment = ee_data['sentiment']
            self.turn = er_data['turn']
        else:
            self.ee_sentence = '[EOP]'
            self.sentiment = [0.0, 0.0, 0.0]
            self.turn = er_data['turn']

    def update_id(self, didx):
        self.id = '%s-%s' % (didx, self.turn)


class PfGDataset(object):
    def __init__(self, jsonl_file):
        reader = jsonlines.open(jsonl_file)

        self.dialogues = [x for x in reader]
        self.data = list()

        for didx, dialogue in enumerate(self.dialogues):
            for idx, utterance in enumerate(dialogue):
                if idx % 2 == 1:
                    continue
                next_utterance = dialogue[idx + 1] if idx + 1 < len(dialogue) \
                    else None
                data_per_turn = PfGData(utterance, next_utterance)
                data_per_turn.update_id(didx)

                self.data.append(data_per_turn)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        current = self.data[index]

        next_strategy = '[EOP]'
        next_connective = '[EOP]'

        if index + 1 < len(self):  # not out of range
            next_one = self.data[index + 1]
            if next_one.turn - current.turn > 0:  # continous
                next_strategy = next_one.strategy
                next_connective = next_one.connective

        return {
            'id': current.id,
            'er_sentence': current.er_sentence,
            'ee_sentence': current.ee_sentence,
            'strategy': current.strategy,
            'sentiment': current.sentiment,
            'connective': current.connective,
            'turn': current.turn,
            'next_strategy': next_strategy,
            'next_connective': next_connective
        }


STRATEGIES = [None, '[EOP]'] + things.STRATEGIES
COMMON_CONS = [None, '[EOP]'] + things.COMMON_CONS


def pfg_features(er_sentence,
                 ee_sentence,
                 strategy,
                 next_strategy,
                 next_connective,
                 tokenizer):
    encoded = tokenizer.encode_plus(er_sentence, ee_sentence)
    next_strategy_idx = STRATEGIES.index(next_strategy)
    next_connective_idx = COMMON_CONS.index(next_connective)
    strategy_changed = 1
    if strategy == next_strategy:
        strategy_changed = 0
    return (encoded['input_ids'],
            encoded['token_type_ids'],
            encoded['attention_mask'],
            next_strategy_idx,
            next_connective_idx,
            strategy_changed)


class PfGFeatureDataset(object):
    def __init__(self,
                 dataset,
                 bert_type,
                 lazy=False,
                 return_sample=False):
        self.dataset = dataset
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_type)
        self.return_sample = return_sample

        if not lazy:
            self.lazy = True
            self.dataset = [self[index] for index in trange(
                0, len(self.dataset), desc='Preprocessing')]
        self.lazy = lazy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if not self.lazy:
            return self.dataset[index]

        sample = self.dataset[index]
        out = pfg_features(sample['er_sentence'],
                           sample['ee_sentence'],
                           sample['strategy'],
                           sample['next_strategy'],
                           sample['next_connective'],
                           self.tokenizer)

        if self.return_sample:
            out = (out, sample)

        return out

    def decode(self, input_ids):
        return self.tokenizer.decode(input_ids)
