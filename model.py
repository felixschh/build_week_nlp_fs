from torch import nn
import torch.nn.functional as F
from torch import optim
import torch
# emb_dim = 300
# class Classifier(nn.Module):
#     def __init__(self, max_seq_len, emb_dim, hidden1=16, hidden2=16):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(max_seq_len*emb_dim, hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.fc3 = nn.Linear(hidden2, 6)
    
    
#     def forward(self, inputs):
#         x = F.relu(self.fc1(inputs.squeeze(1).float()))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


from transformers import BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

bert_model = BertModel.from_pretrained('bert-base-uncased')

class Classifier(nn.Module):
    def __init__(self, bert_model):
        super(Classifier, self).__init__()
        self.emb = bert_model # creating the embeddings (with emb dim == 768)
        self.fc = nn.Linear(768, 6)
    
    def forward(self, ids, mask, token_type_ids):
        output = self.emb(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output = output.pooler_output
        output = self.fc(output)
        return output

model = Classifier(bert_model)