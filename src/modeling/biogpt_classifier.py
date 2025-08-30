import torch
import torch.nn as nn
from transformers import BioGptModel, BioGptPreTrainedModel


class BioGptForSequenceClassification(BioGptPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = BioGptModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Loss function for multi-label classification
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)
        pooled_output = hidden_state[:, -1, :]    # use the last token's hidden state (BioGPT-style)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels.float())

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}