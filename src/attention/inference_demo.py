#%%
from preprocess import read_text
from preprocess import preprocess
from preprocess import make_data
from preprocess import dataloader
import torch.nn as nn
import torch.optim as optim
import torch
from model import BERT, cal_layer_weight, predict, attn_vis

MODEL_PATH = "../../model/BertModel.pth"
batch_size = 32
read_num = 100
MAX_DOC_LEN = 100
vocab_size = 1000
# batch = make_data(doc_list,token_list,word2idx,labels,max_pred,maxlen)
model = BERT(vocab_size)
model.load_state_dict(torch.load(MODEL_PATH))
w_across_layers = cal_layer_weight(model)
# predict(model, batch, 0)
example = "被告的诉请证据不足，本院不予审理。"
attn_vis(model, example, w_across_layers, word2idx, pooling=True)
