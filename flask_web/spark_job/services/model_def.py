import torch
import torch.nn as nn
from transformers import AutoModel

class SanaiChainModel(nn.Module):
    def __init__(self, model_name, num_labels_dict, feature_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.feature_layer = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Dropout(0.1))
        self.base_dim = hidden_size + 64
        self.classifier_pool = nn.Sequential(nn.Linear(self.base_dim, self.base_dim), nn.ReLU(), nn.Dropout(0.1))

        # Stage 1
        self.head_dept = nn.Linear(self.base_dim, num_labels_dict['label_dept'])
        self.head_channel = nn.Linear(self.base_dim, num_labels_dict['mail_channel'])
        self.head_sentiment = nn.Linear(self.base_dim, num_labels_dict['sentiment'])
        self.head_complaint = nn.Linear(self.base_dim, num_labels_dict['is_complaint'])

        # Stage 2
        dim_stage2 = self.base_dim + num_labels_dict['label_dept'] + \
                     num_labels_dict['mail_channel'] + num_labels_dict['sentiment'] + \
                     num_labels_dict['is_complaint']

        self.head_priority = nn.Sequential(
            nn.Linear(dim_stage2, 64), nn.ReLU(),
            nn.Linear(64, num_labels_dict['priority_level'])
        )

        # Stage 3
        dim_stage3 = dim_stage2 + num_labels_dict['priority_level']
        self.head_assignee = nn.Sequential(
            nn.Linear(dim_stage3, 64), nn.ReLU(),
            nn.Linear(64, num_labels_dict['assignee'])
        )

    def forward(self, input_ids, attention_mask, extra_features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.last_hidden_state[:, 0, :]
        feat_output = self.feature_layer(extra_features)
        base_vec = self.classifier_pool(torch.cat((pooler_output, feat_output), dim=1))

        l_dept = self.head_dept(base_vec)
        l_channel = self.head_channel(base_vec)
        l_sentiment = self.head_sentiment(base_vec)
        l_complaint = self.head_complaint(base_vec)

        in_stage2 = torch.cat([base_vec, l_dept, l_channel, l_sentiment, l_complaint], dim=1)
        l_priority = self.head_priority(in_stage2)

        in_stage3 = torch.cat([in_stage2, l_priority], dim=1)
        l_assignee = self.head_assignee(in_stage3)

        return {
            'label_dept': l_dept, 'mail_channel': l_channel, 'sentiment': l_sentiment,
            'is_complaint': l_complaint, 'priority_level': l_priority, 'assignee': l_assignee
        }