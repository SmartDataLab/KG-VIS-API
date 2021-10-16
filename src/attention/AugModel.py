import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import re

MAX_DOC_LEN = 100
maxlen = MAX_DOC_LEN // 2 + 10 # 输入文本的最大词长度，与padding相关
batch_size = 32
max_pred = MAX_DOC_LEN // 2 + 10 # 最大预测词长度，需要padding和mask
n_layers = 6
n_heads = 6
d_k = d_v = 24  # dimension of K(=Q), V
d_model = n_heads * d_v # equal to n_heads * d_v
d_ff = d_model*4 # 4*d_model, FeedForward dimension
lower_dimension = 5
kg_emb_dimension = 5
alpha=0.1
n_graph_heads = 3

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def adj_tanh(x,a):
    m = a * x
    return (torch.exp(m) - torch.exp(-m))/(torch.exp(m) + torch.exp(-m))


class Embedding(nn.Module):
    def __init__(self,vocab_size):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        #self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).cuda()  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x) + self.pos_embed(pos) 
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        #scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        #attn = nn.Softmax(dim=-1)(scores)
        attn = adj_tanh(scores,a=alpha)
        attn.masked_fill(attn_mask,0.0)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layernorm = nn.LayerNorm(d_model)
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
        output = self.linear(context)
        return self.layernorm(output + residual) # output: [batch_size, seq_len, d_model]

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

class GraphAttn(nn.Module):
    def __init__(self):
        super(GraphAttn, self).__init__()
        self. compress = nn.Linear(3,1) ## 三元组变一元

    def forward(self, query, emb_tri):
        #print(emb_tri.size())
        compressed_emb = self.compress(emb_tri.transpose(-1,-2)).squeeze(dim=-1)
        attn = adj_tanh(torch.matmul(query,compressed_emb.transpose(-1,-2))[:,0,:],alpha)
        context = torch.matmul(attn,compressed_emb)[:,0,:]
        return context
    
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
        self.dense1 = nn.Linear(vocab_size,d_model)
        self.dense2 = nn.Linear(d_model*2,2)
        self.graph_layers = nn.ModuleList([GraphAttn() for i in range(n_graph_heads)])
        self.compress = nn.Linear(n_graph_heads,1)
        self.tri_emb_compress = nn.Linear(d_model,2)

    def forward(self, input_ids, masked_pos, emb_tri,corr_tri,status='train'):
        tri_vec = self.embedding.tok_embed(emb_tri)
        compressed_tri_vec = self.tri_emb_compress(tri_vec) # output picture
        
        if status == 'train':
            corr_vec = self.embedding.tok_embed(emb_tri)
            compressed_corr_vec = self.tri_emb_compress(tri_vec) # output picture
        
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
        Query = self.dense1(logits_lm[:,0,:])
            
        contexts = [attn_head(Query,tri_vec) for attn_head in self.graph_layers]
        temp = torch.stack(contexts)
        temp = self.compress(temp.transpose(0,-1)).squeeze(dim=-1)
        
        concat = torch.cat([Query,temp.transpose(0,1)],dim=1)
        logits_pred = self.dense2(concat)
        
        if status == 'train':
            compressed_corr_vec = self.tri_emb_compress(tri_vec) # output picture
            return logits_lm ,logits_pred, Query, compressed_tri_vec, compressed_corr_vec
        else:
            return logits_lm ,logits_pred, Query, compressed_tri_vec

def train(model,loader,epoch_num,Corrupt,criterion,optimizer,vocab_size,param):
    from tqdm import tqdm
    for epoch in tqdm(range(epoch_num)):
        for input_ids, masked_tokens, masked_pos,emb_tri, label in loader:
            corrupted = Corrupt(emb_tri)
            logits_lm,logits_pred,query,tri_vec,corr_vec = model(input_ids, masked_pos,emb_tri,corrupted)
            loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
            loss_lm = (loss_lm.float()).mean()
            loss_pred = criterion(nn.functional.softmax(logits_pred,dim=1), label)
            loss_emb = torch.abs(tri_vec[:,:,0,:] + tri_vec[:,:,2,:] - tri_vec[:,:,1,:]).sum() - \
                        torch.abs(corr_vec[:,:,0,:] + corr_vec[:,:,2,:] - corr_vec[:,:,1,:]).sum()
            loss = loss_lm + loss_pred + param*loss_emb
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
def predict(model,batch):
    # 计算模型预测的准确率
    pred_labels = []
    true_labels = []
    model.eval()
    with torch.no_grad():
        for i in range(len(batch)):
            input_ids, masked_tokens, masked_pos, emb_tri, label = batch[i]
            logits_lm,logits_pred,query,tri_vec = model(torch.LongTensor([input_ids]).cuda(), \
                                                        torch.LongTensor([masked_pos]).cuda(), 
                                                        emb_tri.unsqueeze(dim=0).cuda(),
                                                       None, status='test')
            logits_pred = nn.functional.softmax(logits_pred,dim=1)
            logits_pred = logits_pred.data.max(1)[1][0].tolist()
            pred_labels.append(logits_pred)
            true_labels.append(label)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)

    from mlxtend.plotting import plot_confusion_matrix
    import matplotlib.pyplot as plt

    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

    accuracy = (cm[0][0] + cm[1][1]) / sum(sum(cm))
    print('accuracy:{:.3f}%'.format(accuracy*100))

def code_triple(model,emb_tri_list,eps):
    emb_loss = []
    good_tri_emb = []
    model.eval()
    with torch.no_grad():
        for item in emb_tri_list:
            for tri in item:
                if tri != [0,0,0]:
                    triple_emb = model.tri_emb_compress(model.embedding.tok_embed(torch.tensor(tri).cuda())).cpu().numpy()
                    loss = ((triple_emb[0,:] + triple_emb[2,:] - triple_emb[1,:]) ** 2).mean()
                    emb_loss.append(loss)
                    if loss < eps:
                        #print(tri)
                        if tri not in good_tri_emb:
                            good_tri_emb.append(tri)
    return emb_loss,good_tri_emb

def encode_triple_vis(model,good_tri_emb,good_tri,idx):
    import matplotlib.pyplot as plt
    show_list = []
    model.eval()
    with torch.no_grad():
        for tri in good_tri_emb:
            temp = model.tri_emb_compress(model.embedding.tok_embed(torch.tensor(tri).cuda())).cpu().detach().numpy()
            show_list.append(temp)

    fig = plt.figure(figsize=(6,6))
    A = show_list[idx][0]
    B = show_list[idx][1]
    C = show_list[idx][2]

    plt.rcParams['axes.unicode_minus'] = False
    plt.arrow(0,0,A[0],A[1],length_includes_head = True,head_width = 0.05,head_length = 0.05,fc = 'r',ec = 'b')
    plt.annotate(good_tri[idx][0],((0+A[0])/2,(0+A[1])/2),fontsize=15)

    plt.arrow(A[0],A[1],C[0],C[1] ,length_includes_head = True,head_width = 0.05,head_length = 0.05,fc = 'y',ec = 'g')
    plt.annotate(good_tri[idx][1],((C[0]+2*A[0])/2,(C[1]+2*A[1])/2),fontsize=15)

    plt.arrow(0,0,B[0],B[1],length_includes_head = True,head_width = 0.05,head_length = 0.05,fc = 'r',ec = 'b')
    plt.annotate(good_tri[idx][2],((0+B[0])/2,(0+B[1])/2),fontsize=15)

    #ax.set_xlim(-1.,1.) #设置图形的范围，默认为[0,1]
    #ax.set_ylim(-1.,1.) #设置图形的范围，默认为[0,1]
    plt.grid()  #添加网格
    #ax.set_aspect('equal')  #x轴和y轴等比例
    plt.title('knowledge embedding visualization')
    #plt.tight_layout()
    plt.show()
    
def graph_attn_vis(model,batch,idx,idx2word,n_graph_heads):
    import matplotlib.pyplot as plt
    import numpy as np
    model.eval()
    with torch.no_grad():
        input_ids, masked_tokens, masked_pos, emb_tri, label = batch[idx]
        x_axis = list()
        for tri in emb_tri.cpu().numpy():
            for item in tri:
                x_axis.append(idx2word[item])
        logits_lm,logits_pred,query,com_tri_vec = model(torch.LongTensor([input_ids]).cuda(), \
                                                        torch.LongTensor([masked_pos]).cuda(), \
                                                        emb_tri.unsqueeze(dim=0).cuda(), \
                                                       None,status='test')
        weight_list = []
        for i in range(n_graph_heads):
            tri_vec = model.embedding.tok_embed(emb_tri.cuda())
            compressed_emb = model.graph_layers[i].compress(tri_vec.transpose(-1,-2)).squeeze(dim=-1)
            attn = adj_tanh(torch.matmul(query,compressed_emb.transpose(-1,-2)),alpha)
            token_weight = torch.matmul(attn.transpose(-1,-2),model.graph_layers[i].compress.weight.detach()).view(1,-1)
            weight_list.append(list(token_weight.cpu().numpy()[0]))
        import seaborn as sns
        plt.figure(figsize=(12,6))
        plt.rcParams['axes.unicode_minus'] = False
        sns.heatmap(weight_list,cmap='bwr',annot=True)
        plt.xticks(np.arange(len(x_axis))+0.5, x_axis,rotation=90)
        plt.yticks(np.arange(3)+0.5,['head1','head2','head3'],size=15)
    return weight_list
        
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
    word_embedding = model.embedding(torch.tensor(test_input).unsqueeze(dim=0).cuda()) # batchsize * sequence_len * emb_dim
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
                output = model.layers[i].enc_self_attn.linear(context)
                output = model.layers[i].enc_self_attn.layernorm(output + Q.transpose(1, 2).contiguous().view(1, -1, n_heads * d_v))
                output = model.layers[i].pos_ffn.fc2(gelu(model.layers[i].pos_ffn.fc1(output)))
    res = np.moveaxis(torch.cat(score_list).cpu().detach().numpy(),(0,1,2,3),(3,0,1,2))
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
        return m
    else:
        fig = plt.figure(figsize=(27,18))
        plt.rcParams['axes.unicode_minus'] = False
        n_num = n_heads
        for i in range(n_num):
            ax = fig.add_subplot(n_num // 3,3,i+1)
            sns.heatmap(final[i,:,:],annot=True,cmap='bwr')
            plt.xticks(range(len(token_input)), token_input,rotation=90,size=10)
            plt.yticks(range(len(token_input)), token_input,rotation=360,size=10)
        return final[i,:,:]