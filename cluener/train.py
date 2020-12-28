import numpy as np
from tqdm import tqdm
import random
import torch
from transformers import BertTokenizer, BertConfig, BertModel
import torch.optim as optim
import torch.nn as nn
from torchtext import data
from net import BiLSTMCRF, BERTCRF, BERTBiLSTMCRF, BERTransformerCRF
from forward_step import *
import pickle

gpu = 1
device = torch.device('cuda:%d' % (
    gpu) if torch.cuda.is_available() else 'cpu')

####################################################################
###################### Load data & preprocess ######################
####################################################################
# 'bert-base-chinese' 'hfl/chinese-roberta-wwm-ext'
PRETRAINED_MODEL_NAME = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

with open('./token_info/ner2int.pickle', 'rb') as file:
    ner2int = pickle.load(file)
file.close()


def str_split(x):
    return x.split(' ')


def ner_onehot(x):
    return [ner2int[k] for k in x]


CONTENT = data.Field(sequential=True, tokenize=str_split, preprocessing=tokenizer.convert_tokens_to_ids, pad_token=tokenizer.pad_token_id,
                     unk_token=tokenizer.unk_token_id, fix_length=50, use_vocab=False, lower=True, batch_first=True, include_lengths=False)
NER = data.Field(sequential=True, tokenize=str_split, preprocessing=ner_onehot, pad_token=len(
    ner2int), fix_length=50, use_vocab=False, lower=False, batch_first=True, include_lengths=False)

train_data, valid_data = data.TabularDataset.splits(
    path='./data',
    train='cluener_train_v2.csv',
    validation='cluener_valid_v2.csv',
    format='csv',
    fields={'content': ('content', CONTENT), 'ner': ('ner', NER)}
)

########################################################
###################### Model info ######################
########################################################

BATCH_SIZE = 32

train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort=False
    # sort_within_batch=True,
    # sort_key=lambda x: len(x.content)
)

bert_config = BertConfig.from_pretrained(PRETRAINED_MODEL_NAME)
vocab_size = bert_config.vocab_size
embedding_dim = bert_config.pooler_fc_size
num_class = len(ner2int)
# model = BiLSTMCRF(vocab_size, embedding_dim, num_class+1)
# word_emb = BertModel.from_pretrained(PRETRAINED_MODEL_NAME).embeddings.word_embeddings.weight.data.to(device)
# model.embedding.weight.data.copy_(word_emb)
model = BERTCRF(bert_config, embedding_dim, num_class+1)
# model = BERTransformerCRF(bert_config, embedding_dim, num_class+1, 'cuda:%d'%(gpu) if torch.cuda.is_available() else 'cpu')
optimizer = optim.Adam(model.parameters(), lr=1e-05)
model.to(device)

############################################################
###################### Start training ######################
############################################################

EPOCHS = 200
EARLY_STOP = 8
best_val_loss = 1e10
best_val_accn = 0
early_count = 0
len_train, len_valid = len(train_iterator), len(valid_iterator)
save_path = './model/'
model_name = 'bert-ner'
status = '_best_test_v2'
for epoch in range(EPOCHS):
    if epoch == 5:
        model.freeze_bert()
        optimizer = optim.Adam(model.parameters(), lr=5e-04)
    elif epoch == 10:
        model.unfreeze_bert()
        optimizer = optim.Adam(model.parameters(), lr=5e-06)
    elif epoch == 12:
        model.freeze_bert()
        optimizer = optim.Adam(model.parameters(), lr=5e-04)
    batch_train_loss, batch_train_acc, batch_val_loss, batch_val_correct, batch_val_total, batch_val_correctn, batch_val_totaln = [], [], [], [], [], [], []
    train_loop = tqdm(enumerate(train_iterator), total=len_train)
    for batch_idx, batch in train_loop:
        train_loss, train_acc = training_step(batch, model, optimizer, device)
        batch_train_loss.append(train_loss)
        batch_train_acc.append(train_acc)
        epoch_idx = '{0:02d}'.format(epoch+1)
        train_loop.set_description(f'Epoch [{epoch_idx}/{EPOCHS}] ')
        train_loop.set_postfix(loss=train_loss, acc=train_acc)
        if batch_idx == len(train_iterator)-1:
            valid_loop = tqdm(enumerate(valid_iterator),
                              total=len_valid, position=0, leave=False)
            for batch_idx, batch in valid_loop:
                valid_loss, val_correct, val_total, val_correctn, val_totaln = validation_step(
                    batch, model, device)
                batch_val_loss.append(valid_loss)
                batch_val_correct.append(val_correct)
                batch_val_total.append(val_total)
                batch_val_correctn.append(val_correctn)
                batch_val_totaln.append(val_totaln)
                valid_loop.set_description(
                    f'Epoch [{epoch+1}/{EPOCHS}] : Validating... [{batch_idx+1}/{len_valid}]')
            sub_batch = int(len_train*0.05)
            avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc, avg_val_accn = np.mean(batch_train_loss[-sub_batch:]), np.mean(batch_val_loss), np.mean(batch_train_acc), np.round(
                (np.sum(batch_val_correct)/np.sum(batch_val_total)).detach().cpu().numpy(), 5), np.round((np.sum(batch_val_correctn)/np.sum(batch_val_totaln)).detach().cpu().numpy(), 5)
            train_loop.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss,
                                   train_acc=avg_train_acc, val_acc=avg_val_acc, val_accn=avg_val_accn)
    ##############################
    ##### Save log and model #####
    ##############################
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_count = 0
        torch.save(model.state_dict(), save_path+model_name+status+'.ckpt')
    else:
        early_count += 1
    if early_count == EARLY_STOP:
        print('\nEarly stop\n')
        break
