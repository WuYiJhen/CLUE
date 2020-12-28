import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from transformers import BertPreTrainedModel, BertModel
from crf.crf import ConditionalRandomField
import pickle
# hfl/chinese-roberta-wwm-ext bert-base-chinese
PRETRAINED_MODEL_NAME = 'bert-base-chinese'
with open('./token_info/ner2int.pickle', 'rb') as file:
    ner2int = pickle.load(file)
file.close()

idx2tag = {}
for k in ner2int:
    idx2tag[ner2int[k]] = k


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        x = x * nn.Sigmoid()(x)
        return x


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_class, dropout=0.2):
        super(BiLSTMCRF, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.Y = num_class
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            self.embedding_dim, self.embedding_dim // 2, num_layers=3, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout)
        self.h2 = nn.Linear(self.embedding_dim, self.Y)
        xavier_uniform_(self.h2.weight)
        # CRF
        self.crf = ConditionalRandomField(self.Y,
                                          label_encoding="BIO",
                                          idx2tag=idx2tag)

    def forward(self, batch, crf_mask=None):
        x = self.embedding(batch.content)
        x, _ = self.lstm(x.transpose(0, 1))
        x = self.dropout(x)
        x = self.h2(x)  # (S, N, Y)
        log_likelihood = self.crf(x.transpose(0, 1), batch.ner, crf_mask)
        loss = -log_likelihood
        pred = self.crf.best_viterbi_tag(x.transpose(0, 1), crf_mask)
        pred = [x[0][0] for x in pred]
        return loss, pred


class BERTCRF(BertPreTrainedModel):
    def __init__(self, conf, embedding_dim, num_class, dropout=0.2):
        super(BERTCRF, self).__init__(conf)
        self.embedding_dim = embedding_dim
        self.Y = num_class
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden = nn.Linear(self.embedding_dim, self.Y)
        xavier_uniform_(self.hidden.weight)
        # CRF
        self.crf = ConditionalRandomField(self.Y,
                                          label_encoding="BIO",
                                          idx2tag=idx2tag)

    def key_padding_mask_bert(self, x, pad_num=0):
        atten_mask = x.masked_fill(x != pad_num, 1)
        return atten_mask

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, batch, crf_mask=None, inference=False):
        x = self.bert(input_ids=batch.content, attention_mask=self.key_padding_mask_bert(
            batch.content))[0]  # (N, S, D)
        x = self.dropout(x)
        x = self.hidden(x)  # (N, S, Y)
        if not inference:
            log_likelihood = self.crf(x, batch.ner, crf_mask)
            loss = -log_likelihood
            pred = self.crf.best_viterbi_tag(x, crf_mask)
            pred = [x[0][0] for x in pred]
            # x = x.reshape(x.shape[0]*x.shape[1], x.shape[2]) # (N*S, Y)
            # y = batch.ner.reshape(-1) # (N*S)
            # idx1 = y != 21
            # x = x[idx1]
            # y = y[idx1]
            # loss = nn.CrossEntropyLoss()(x, y)
            # pred = torch.argmax(x, dim=1)
            return loss, pred
        else:
            pred = self.crf.best_viterbi_tag(x, crf_mask)
            pred = [x[0][0] for x in pred]
            return pred


class BERTBiLSTMCRF(BertPreTrainedModel):
    def __init__(self, conf, embedding_dim, num_class, dropout=0.2):
        super(BERTBiLSTMCRF, self).__init__(conf)
        self.embedding_dim = embedding_dim
        self.Y = num_class
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        self.lstm = nn.LSTM(
            self.embedding_dim, self.embedding_dim // 2, num_layers=3, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden = nn.Linear(self.embedding_dim, self.Y)
        xavier_uniform_(self.hidden.weight)
        # CRF
        self.crf = ConditionalRandomField(self.Y,
                                          label_encoding="BIO",
                                          idx2tag=idx2tag)

    def key_padding_mask_bert(self, x, pad_num=0):
        atten_mask = x.masked_fill(x != pad_num, 1)
        return atten_mask

    def forward(self, batch, crf_mask=None):
        x = self.bert(input_ids=batch.content, attention_mask=self.key_padding_mask_bert(
            batch.content))[0]  # (N, S, D)
        x, _ = self.lstm(x.transpose(0, 1))
        x = x.transpose(0, 1)
        x = self.dropout(x)
        x = self.hidden(x)  # (N, S, Y)
        log_likelihood = self.crf(x, batch.ner, crf_mask)
        loss = -log_likelihood
        pred = self.crf.best_viterbi_tag(x, crf_mask)
        pred = [x[0][0] for x in pred]
        return loss, pred


class BERTransformerCRF(BertPreTrainedModel):
    def __init__(self, conf, embedding_dim, num_class, device, dropout=0.2):
        super(BERTransformerCRF, self).__init__(conf)
        self.embedding_dim = embedding_dim
        self.Y = num_class
        self.dev = device
        self.bert = BertModel.from_pretrained(PRETRAINED_MODEL_NAME)
        encoder_layer = nn.TransformerEncoderLayer(
            self.embedding_dim, nhead=12, dim_feedforward=1024)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden = nn.Linear(self.embedding_dim, self.Y)
        xavier_uniform_(self.hidden.weight)
        # CRF
        self.crf = ConditionalRandomField(self.Y,
                                          label_encoding="BIO",
                                          idx2tag=idx2tag)

    def key_padding_mask_bert(self, x, pad_num=0):
        atten_mask = x.masked_fill(x != pad_num, 1)
        return atten_mask

    def sliding_atten_mask(self, x, window=5):
        atten_mask = torch.ones(x.shape[1], x.shape[1])
        atten_mask = atten_mask.float().masked_fill(atten_mask == 1, float('-inf'))
        for i in range(atten_mask.shape[0]):
            if i < window:
                atten_mask[i][:(i+1+window)] = 0
            else:
                atten_mask[i][(i-window):(i+1+window)] = 0
        return atten_mask.to(torch.device(self.dev))

    def key_padding_mask(self, x):
        padding_mask = x == 0
        return padding_mask

    def forward(self, batch, crf_mask=None, inference=False):
        x = self.bert(input_ids=batch.content, attention_mask=self.key_padding_mask_bert(
            batch.content))[0]  # (N, S, D)
        x = self.encoder(x.transpose(0, 1), mask=self.sliding_atten_mask(
            batch.content)).transpose(0, 1)
        x = self.dropout(x)
        x = self.hidden(x)  # (N, S, Y)
        log_likelihood = self.crf(x, batch.ner, crf_mask)
        loss = -log_likelihood
        pred = self.crf.best_viterbi_tag(x, crf_mask)
        pred = [x[0][0] for x in pred]
        return loss, pred
