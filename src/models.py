import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

import things


class BertForPersuasiveConnection(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_for_connective = nn.Linear(
            config.hidden_size, len(things.COMMON_CONS) + 2)
        self.linear_for_strategy = nn.Linear(
            config.hidden_size, len(things.STRATEGIES) + 2)
        self.linear_for_changing = nn.Linear(config.hidden_size, 2)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = self.dropout(outputs[1])

        pred_strt = self.linear_for_strategy(pooled_output)
        pred_conn = self.linear_for_connective(pooled_output)
        pred_chng = self.linear_for_changing(pooled_output)

        return pred_strt, pred_conn, pred_chng
