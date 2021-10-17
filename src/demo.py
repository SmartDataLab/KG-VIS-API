"""
demo
"""
import os
from typing import Optional
from fastapi import FastAPI
from dotenv import load_dotenv  # for python-dotenv method
from py2neo import Graph
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

load_dotenv("neo.env")  # for python-dotenv method


password = os.environ.get("neo_password")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
graph = Graph("bolt://localhost:7687", auth=("neo4j", password))


@app.get("/")
def read_root():
    """[summary]

    Returns:
        [type]: [description]
    """
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q_option: Optional[str] = None):
    """[summary]

    Returns:
        [type]: [description]
    """
    return {"item_id": item_id, "q": q_option}


def process_output(data: dict):
    """[summary]

    Args:
        data (dict): [description]

    Returns:
        [type]: [description]
    """
    cate2idx = {"c": 0, "f": 1, "a": 2, "k": 3, "o": 4}
    categories = [
        {"name": "CASE"},
        {"name": "FACT"},
        {"name": "ACTION"},
        {"name": "KEYWORD"},
        {"name": "OBJECT"},
    ]
    nodes = []
    links = []
    for one in data:
        for key, value_dict in one.items():
            if key == "node":
                node = value_dict.copy()
                for attr_name, attr_value in value_dict.items():
                    if "id" in attr_name:
                        id_name = attr_name
                        id_value = int(attr_value)
                        node["id"] = id_value + cate2idx[id_name.split("_")[0]] * 1000
                        node["category"] = cate2idx[id_name.split("_")[0]]
                    else:
                        node["name"] = attr_value
                nodes.append(node)
            else:
                link = {"source": "0", "target": "1"}
                links.append(link)
    return {"nodes": nodes, "links": links, "categories": categories}


@app.get("/neo/{label}")
def cypher_result(
    label: str,
    limit: Optional[int] = 300,
    id: Optional[int] = 19,
    maxDepth: Optional[int] = 3,
    type_id_str: Optional[str] = "c_id",
):
    """[summary]

    Args:
        label (str): [description]
        limit (Optional[int], optional): [description]. Defaults to 300.
        id (Optional[int], optional): [description]. Defaults to 19.
        maxDepth (Optional[int], optional): [description]. Defaults to 3.
        type_id_str (Optional[str], optional): [description]. Defaults to "c_id".

    Returns:
        [type]: [description]
    """
    res = graph.run(
        f"MATCH (a:{label}"
        + " {"
        + type_id_str
        + ':"'
        + str(id)
        + '"}'
        + ") WITH id(a) AS startNode CALL gds.alpha.bfs.stream('myGraph', {"
        + f"startNode: startNode, maxDepth: {maxDepth}"
        + "}"
        + f") YIELD path UNWIND [ n in nodes(path) | n ] AS node RETURN node LIMIT {limit}"
    ).data()
    js_data = jsonable_encoder(process_output(res))
    return JSONResponse(js_data)


# import package for interpretability
import torch
import jieba
import numpy as np
import pickle
import torch.nn as nn
import math
from attention import model
from attention.model import BERT
from load_model import get_model

id2word = pickle.load(open("../model/idx2word_v2.pickle", "rb"))
word2idx = {value: key for key, value in id2word.items()}
MODEL_PATH = "../model/BertModel.pth"
MODEL_PATH = "../model/AugModel.pth"
batch_size = 32
read_num = 100
MAX_DOC_LEN = 100
vocab_size = len(word2idx)
#%%
model2 = get_model(MODEL_PATH)
# batch = make_data(doc_list,token_list,word2idx,labels,max_pred,maxlen)
# model2 = BERT(vocab_size)
# model2.load_state_dict(torch.load(MODEL_PATH))
# model2 = torch.load(MODEL_PATH)
# predict(model, batch, 0)
model2 = model2.to(torch.device("cpu"))
#%%
# w_across_layers = cal_layer_weight(model)
w_across_layers = torch.tensor([0.8335, 0.1133, 0.0532])
w_across_layers = torch.tensor([0.2268, 0.2608, 0.1906, 0.1782, 0.0929, 0.0506])

# %%
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def adj_tanh(x, a):
    m = a * x
    return (torch.exp(m) - torch.exp(-m)) / (torch.exp(m) + torch.exp(-m))


def attn_vis(model, example, w, word2idx, pooling=True):
    MAX_DOC_LEN = 100
    maxlen = MAX_DOC_LEN // 2 + 8  # 输入文本的最大词长度，与padding相关
    batch_size = 32
    max_pred = MAX_DOC_LEN // 2 + 8  # 最大预测词长度，需要padding和mask
    n_layers = 3
    n_heads = 12
    d_model = 768
    d_ff = 768 * 4  # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_segments = 2
    alpha = 0.3

    MAX_DOC_LEN = 100
    maxlen = MAX_DOC_LEN // 2 + 10  # 输入文本的最大词长度，与padding相关
    batch_size = 32
    max_pred = MAX_DOC_LEN // 2 + 10  # 最大预测词长度，需要padding和mask
    n_layers = 6
    n_heads = 6
    d_k = d_v = 24  # dimension of K(=Q), V
    d_model = n_heads * d_v  # equal to n_heads * d_v
    d_ff = d_model * 4  # 4*d_model, FeedForward dimension
    lower_dimension = 5
    kg_emb_dimension = 5
    alpha = 0.1
    n_graph_heads = 3
    token_input = ["[CLS]"] + list(jieba.cut(example))
    test_input = [word2idx[item] for item in token_input]
    word_embedding = model.embedding(torch.tensor(test_input).unsqueeze(dim=0))
    K = model.W_K(word_embedding).view(1, -1, n_heads, d_k).transpose(1, 2)
    V = model.W_V(word_embedding).view(1, -1, n_heads, d_k).transpose(1, 2)

    model.eval()
    output = word_embedding
    score_list = []
    with torch.no_grad():
        for i in range(n_layers):
            Q = (
                model.layers[i]
                .enc_self_attn.W_Q(output)
                .view(1, -1, n_heads, d_k)
                .transpose(1, 2)
            )
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
            score_list.append(scores)
            if i == n_layers - 1:
                break
            else:
                attn = adj_tanh(scores, alpha)
                context = torch.matmul(attn, V)
                context = (
                    context.transpose(1, 2).contiguous().view(1, -1, n_heads * d_v)
                )  # context: [batch_size, seq_len, n_heads, d_v]
                output = nn.Linear(n_heads * d_v, d_model)(context)
                output = nn.LayerNorm(d_model)(
                    output + Q.transpose(1, 2).contiguous().view(1, -1, n_heads * d_v)
                )
                output = model.layers[i].pos_ffn.fc2(
                    gelu(model.layers[i].pos_ffn.fc1(output))
                )
    res = np.moveaxis(
        torch.cat(score_list).detach().numpy(), (0, 1, 2, 3), (3, 0, 1, 2)
    )
    final = torch.matmul(torch.from_numpy(res), w)
    m = list()
    R = final.size(1)
    L = final.size(2)
    score_matrix = final.detach().numpy()
    for i in range(R):
        row = list()
        for j in range(L):
            index = torch.abs(final).argmax(dim=0)[i][j]
            row.append(float(score_matrix[index][i][j]))
        m.append(row)
    return {"token_input": token_input, "heat_matrix": m}


def cor_map(value, factor=10):
    return round(value * factor)


def process_heatmap_data(matrix_list_data):
    data = []
    for i, row in enumerate(matrix_list_data):
        for j, one in enumerate(row):
            data.append([i, j, cor_map(one, 30)])
    return data


def process_graph_link_data(matrix_list_data, token_input, threshold):
    nodes = []
    links = []
    for i, node in enumerate(token_input):
        nodes.append({"id": str(i), "name": node})
    for i, row in enumerate(matrix_list_data):
        for j, one in enumerate(row):
            if abs(one) > threshold:
                if one > 0:
                    links.append(
                        {"source": str(i), "target": str(j), "value": abs(one)}
                    )
    return {"nodes": nodes, "links": links}


@app.get("/bert_heat/{example}")
def cypher_result(
    example: str,
    threshold: Optional[float] = 0.1,
):
    # example = "被告的诉请证据不足，本院不予审理。"
    data = attn_vis(model2, example, w_across_layers, word2idx)
    data["echart_data"] = process_heatmap_data(data["heat_matrix"])
    data["echart_graph_data"] = process_graph_link_data(
        data["heat_matrix"], data["token_input"], threshold
    )
    return data
