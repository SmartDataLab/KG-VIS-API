import torch.nn as nn
import torch.utils.data as Data
import torch
import numpy as np
import math

# BERT Parameters
MAX_DOC_LEN = 100
maxlen = MAX_DOC_LEN // 2 + 8 # 输入文本的最大词长度，与padding相关
batch_size = 32
max_pred = MAX_DOC_LEN // 2 + 8 # 最大预测词长度，需要padding和mask
n_layers = 3
n_heads = 12
d_model = 768
d_ff = 768*4 # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2
alpha = 0.3

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos,label):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.label = label
  
    def __len__(self):
        return len(self.input_ids)
  
    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.label[idx]

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def adj_tanh(x,a):
    m = a * x
    return (torch.exp(m) - torch.exp(-m))/(torch.exp(m) + torch.exp(-m))

class Embedding(nn.Module):
    def __init__(self,vocab_size):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x) + self.pos_embed(pos) 
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        attn = adj_tanh(scores,a=alpha)
        attn.masked_fill(attn_mask,0.0)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = K
        v_s = V

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual) # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, K, V, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, K, V, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

class BERT(nn.Module):
    def __init__(self,vocab_size):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight
        self.dense = nn.Linear(vocab_size,2)

    def forward(self, input_ids, masked_pos):
        output = self.embedding(input_ids) # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids) # [batch_size, maxlen, maxlen]
        batch_size = output.size(0)
        K = self.W_K(output).view(batch_size, -1, n_heads, d_k).transpose(1,2) 
        V = self.W_V(output).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, K, V, enc_self_attn_mask)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked)) # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]
        logits_pred = self.dense(logits_lm[:,0,:])
        logits_pred = nn.functional.softmax(logits_pred,dim=1)
        return logits_lm,logits_pred

def train(model,loader,epoch,criterion,optimizer):
    from tqdm import tqdm
    for epoch in tqdm(range(epoch)):
        for input_ids, masked_tokens, masked_pos, label in loader:
            logits_lm, logits_pred = model(input_ids, masked_pos)
            loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
            loss_lm = (loss_lm.float()).mean()
            loss_pred = criterion(logits_pred, label)
            loss = loss_lm + loss_pred
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def predict(model,batch,index):
    input_ids, masked_tokens, masked_pos, label = batch[index]
    print('================================')

    logits_lm, logits_pred = model(torch.LongTensor([input_ids]), \
                     torch.LongTensor([masked_pos]))

    logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
    print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
    print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

    logits_pred = logits_pred.data.max(1)[1][0].data.numpy()
    print('label : ', True if label else False)
    print('predict label : ',True if logits_pred else False)
    
def accuracy(model,batch):
    # 计算模型预测的准确率
    pred_labels = []
    true_labels = []
    for i in range(len(batch)):
        input_ids, masked_tokens, masked_pos, label = batch[i]
        logits_lm, logits_pred = model(torch.LongTensor([input_ids]), \
                     torch.LongTensor([masked_pos]))
        logits_pred = logits_pred.data.max(1)[1][0].tolist()
        pred_labels.append(logits_pred)
        true_labels.append(label)
    return pred_labels,true_labels

def cm_plot(pred_labels,true_labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)
    from mlxtend.plotting import plot_confusion_matrix
    import matplotlib.pyplot as plt
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('50 Epochs with alpha equals 0.3', fontsize=18)
    plt.show()
    accuracy = (cm[0][0] + cm[1][1]) / sum(sum(cm))
    print('accuracy:{:.3f}%'.format(accuracy*100))
    
def cal_layer_weight(model):
    import numpy as np
    w_layers = list()
    for i in range(len(model.layers)):
        target_weight = model.layers[i].enc_self_attn.W_Q.weight
        w_layers.append(torch.sum(torch.abs(target_weight.grad)) 
                        / np.sqrt(target_weight.size(0)) )
    w_across_layers = nn.Softmax(dim=-1)(torch.tensor(w_layers))
    return w_across_layers

def attn_vis(model,example,w,word2idx,pooling=True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import jieba
    token_input = ['[CLS]'] + list(jieba.cut(example))
    test_input = [word2idx[item] for item in token_input]
    word_embedding = model.embedding(torch.tensor(test_input).unsqueeze(dim=0))
    K = model.W_K(word_embedding).view(1, -1, n_heads, d_k).transpose(1,2)
    V = model.W_V(word_embedding).view(1, -1, n_heads, d_k).transpose(1,2)

    model.eval()
    output = word_embedding
    score_list = []
    with torch.no_grad():
        for i in range(n_layers):
            Q = model.layers[i].enc_self_attn.W_Q(output).view(1, -1, n_heads, d_k).transpose(1,2)
            scores = (torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k))
            score_list.append(scores)
            if i == n_layers-1:
                break
            else:
                attn = adj_tanh(scores,alpha)
                context = torch.matmul(attn, V)
                context = context.transpose(1, 2).contiguous().view(1, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads, d_v]
                output = nn.Linear(n_heads * d_v, d_model)(context)
                output = nn.LayerNorm(d_model)(output + Q.transpose(1, 2).contiguous().view(1, -1, n_heads * d_v))
                output = model.layers[i].pos_ffn.fc2(gelu(model.layers[i].pos_ffn.fc1(output)))
    res = np.moveaxis(torch.cat(score_list).detach().numpy(),(0,1,2,3),(3,0,1,2))
    final = torch.matmul(torch.from_numpy(res),w)
    if pooling:
        m = list()
        R = final.size(1)
        L = final.size(2)
        score_matrix = final.detach().numpy()
        for i in range(R):
            row = list()
            for j in range(L):
                index = torch.abs(final).argmax(dim=0)[i][j]
                row.append(score_matrix[index][i][j])
            m.append(row)

        plt.figure(figsize=(8,6))
        plt.rcParams['axes.unicode_minus'] = False
        sns.heatmap(m,annot=True,cmap='bwr')
        plt.xticks(np.arange(len(token_input))+0.5, token_input,rotation=90,size=10)
        plt.yticks(np.arange(len(token_input))+0.5, token_input,rotation=360,size=10,va='center')
    else:
        fig = plt.figure(figsize=(27,18))
        plt.rcParams['axes.unicode_minus'] = False
        n_num = n_heads
        for i in range(n_num):
            ax = fig.add_subplot(n_num // 3,3,i+1)
            sns.heatmap(final[i,:,:],annot=True,cmap='bwr')
            plt.xticks(range(len(token_input)), token_input,rotation=90,size=10)
            plt.yticks(range(len(token_input)), token_input,rotation=360,size=10)