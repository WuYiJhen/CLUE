import torch
import pickle
import torch.nn as nn
from itertools import chain
from transformers import BertTokenizer
# 'bert-base-chinese' 'hfl/chinese-roberta-wwm-ext'
PRETRAINED_MODEL_NAME = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

with open('./token_info/ner2int.pickle', 'rb') as file:
    ner2int = pickle.load(file)
file.close()

# For nn.Transformer


def key_padding_mask(x, device):
    padding_mask = x == tokenizer.pad_token_id
    return padding_mask.to(device)

# for pretrained bert


def key_padding_mask_bert(x, device):
    atten_mask = torch.zeros(x.shape, dtype=torch.uint8).to(device)
    atten_mask = atten_mask.masked_fill(x != tokenizer.pad_token_id, 1)
    return atten_mask

# For nn.Transformer


def attention_mask(x, device):
    atten_mask = (torch.triu(torch.ones(x.shape[1], x.shape[1])) == 1)
    atten_mask = atten_mask.float().masked_fill(atten_mask == 0, float(
        '-inf')).masked_fill(atten_mask == 1, float(0.0)).transpose(0, 1).to(device)
    return atten_mask


def categorical_accuracy(preds, y, mod='train'):
    correct = preds.eq(y)
    if mod == 'valid':
        return correct.sum(), y.shape[0]
    else:
        return (correct.sum() / torch.FloatTensor([y.shape[0]])).numpy()[0]


def training_step(batch, model, optimizer, device):
    model.train()
    optimizer.zero_grad()
    crf_mask = key_padding_mask_bert(batch.content, device)
    # crf
    label = batch.ner.reshape(-1)
    loss, pred = model(batch, crf_mask)
    pred = list((chain(*pred)))
    pred = torch.tensor(pred).to(device)
    idx1 = label != len(ner2int)
    idx2 = label != ner2int['O']
    idx = idx1*idx2
    acc = categorical_accuracy(pred, label[idx1])
    loss.backward()
    optimizer.step()
    return loss.item(), acc


def validation_step(batch, model, device):
    model.eval()
    crf_mask = key_padding_mask_bert(batch.content, device)
    # crf
    loss, pred = model(batch, crf_mask)
    label = batch.ner.reshape(-1)
    pred = list((chain(*pred)))
    pred = torch.tensor(pred).to(device)
    idx1 = label != len(ner2int)
    correct, total = categorical_accuracy(pred, label[idx1], 'valid')
    label = label[idx1]
    idx2 = label != ner2int['O']
    try:
        correctn, totaln = categorical_accuracy(
            pred[idx2], label[idx2], 'valid')
    except:
        correctn, totaln = 0, 0
    return loss.item(), correct, total, correctn, totaln
