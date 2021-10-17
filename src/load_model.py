#%%
import torch


def get_model(model_path):
    # MODEL_PATH = "/data1/su/app/knowledge_graph/KG-VIS-API/model/AugModel.pth"
    # batch = make_data(doc_list,token_list,word2idx,labels,max_pred,maxlen)
    # model = BERT(vocab_size)
    # model.load_state_dict(torch.load(MODEL_PATH))
    model2 = torch.load(model_path)
    return model2


# %%
if __name__ == "__main__":
    get_model("../model/AugModel.pth")