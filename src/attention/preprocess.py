## functions for preprocess
import torch.utils.data as Data

def read_text(filename,read_num,MAX_DOC_LEN):
    import re
    doc_list = []
    labels = []
    with open(filename,'r') as f:
        fraudNum = 0
        nonfraudNum = 0
        while fraudNum < read_num or nonfraudNum < read_num: 
            line = f.readline()
            line = line.strip('\n')
            label = int(line[-1])
            if label == 0:
                if nonfraudNum > read_num:
                    continue
                else:
                    nonfraudNum += 1
            else:
                if fraudNum > read_num:
                    continue
                else:
                    fraudNum += 1

            labels.append(label)
            line = line[:-2].strip('\t')
            line = re.sub("[（）？！]", '', line) # "[。，！？\\-]"

            if len(line) <= MAX_DOC_LEN:
                doc_list.append(line)
            else:
                first = line[:MAX_DOC_LEN//2]
                last = line[-MAX_DOC_LEN//2:]
                doc_list.append(first + " " + last)

    f.close()
    return doc_list,labels

def preprocess(doc_list):
    import jieba
    word_list = []
    for doc in doc_list:
        temp = list(jieba.cut(doc))
        for item in temp:
            if item not in word_list:
                word_list.append(item)
    word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[MASK]' : 2}
    for i, w in enumerate(word_list):
        word2idx[w] = i + 3
    idx2word = {i: w for i, w in enumerate(word2idx)}

    token_list = list()
    for doc in doc_list:
        arr = [word2idx[s] for s in list(jieba.cut(doc))]
        token_list.append(arr)
        
    return word_list,word2idx,token_list

def make_data(doc_list,token_list,word2idx,labels,max_pred,maxlen):
    from random import shuffle,random,randint
    vocab_size = len(word2idx)
    batch = []
    for ids in range(len(doc_list)):
        input_ids = [word2idx['[CLS]']] + token_list[ids]
        n_pred =  min(max_pred, max(1, int(len(input_ids) * 0.15))) # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]']] # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos) # 被遮盖单词的位置
            masked_tokens.append(input_ids[pos]) # 被遮盖单词的tokens
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]'] # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                while index < 3: # can't involve 'CLS', 'MASK', 'PAD'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index # replace
        
        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            
        batch.append([input_ids,masked_tokens, masked_pos,labels[ids]])
        
    return batch

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
    
def dataloader(batch,batch_size):
    import torch
    import torch
    import torch.utils.data as Data

    input_ids, masked_tokens, masked_pos, label = zip(*batch)
    input_ids, masked_tokens, masked_pos, label = \
        torch.LongTensor(input_ids),  torch.LongTensor(masked_tokens),\
        torch.LongTensor(masked_pos), torch.LongTensor(label)

    loader = Data.DataLoader(MyDataSet(input_ids,masked_tokens, masked_pos, label), batch_size, True)
    return loader