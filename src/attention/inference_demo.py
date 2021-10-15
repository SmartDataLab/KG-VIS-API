#%%
from preprocess import read_text
from preprocess import preprocess
from preprocess import make_data
from preprocess import dataloader
import torch.nn as nn
import torch.optim as optim
import torch
from model import BERT, cal_layer_weight, predict, attn_vis
import pickle

word2idx = pickle.load(open("../../model/word2idx.pickle", "rb"))
MODEL_PATH = "../../model/BertModel.pth"
batch_size = 32
read_num = 100
MAX_DOC_LEN = 100
vocab_size = len(word2idx)
#%%
# batch = make_data(doc_list,token_list,word2idx,labels,max_pred,maxlen)
# model = BERT(vocab_size)
# model.load_state_dict(torch.load(MODEL_PATH))
model2 = torch.load(MODEL_PATH)
# predict(model, batch, 0)
#%%
# w_across_layers = cal_layer_weight(model)
w_across_layers = torch.tensor([0.8335, 0.1133, 0.0532])

# %%
example = "被告的诉请证据不足，本院不予审理。"
attn_vis(model2, example, w_across_layers, word2idx, pooling=True)
# %%
# %%
# TODO(sujinhua): find the echart graph for the hot map : done
# TODO(sujinhua): fit the data for the option: done
# TODO(sujinhua): build a fastapi: done
# TODO(sujinhua): optional: change into the onnx format
