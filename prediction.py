
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("./")

import io
import csv
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from gensim.models import word2vec
from deep_net import unitNET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def file_input_csv(filename,index_col = None):
    data = pd.read_csv(filename, index_col = index_col)

    return data

def create_vector(data, index, model, num = 4):
    seq_dict = {}

    seq = list(set(data[index].values.tolist()))

    for i in range(len(seq)):
        seq_dict[seq[i]] = [model[seq[i][j:j + num]]  for j in range(len(seq[i]) - num + 1)]

    return seq_dict

def make_vectors(in_path, model_path, num = 4):
    
    data_sets = file_input_csv(in_path, index_col = None)
    
    wv_model = word2vec.Word2Vec.load(model_path)

    human_seq_dict = create_vector(data_sets, "human_seq", wv_model, num = num)
    virus_seq_dict = create_vector(data_sets, "virus_seq", wv_model, num = num)

    return data_sets, human_seq_dict, virus_seq_dict


class pv_data_sets(data.Dataset):
    def __init__(self,data_sets, human_seq_dict, virus_seq_dict):
        super().__init__()
        self.human_seq = data_sets["human_seq"].values.tolist()
        self.virus_seq = data_sets["virus_seq"].values.tolist()
        self.human_seq_dict = human_seq_dict
        self.virus_seq_dict = virus_seq_dict

    def __len__(self):
        return len(self.human_seq)

    def __getitem__(self, idx):
        human_mat = self.human_seq_dict[self.human_seq[idx]]
        virus_mat = self.virus_seq_dict[self.virus_seq[idx]]
        
        return torch.tensor(human_mat), torch.tensor(virus_mat)

def collate_fn(batch):
    human_mat, virus_mat = list(zip(*batch))    

    human_mat_list = torch.nn.utils.rnn.pack_sequence(list(human_mat), enforce_sorted=False).to(device).float()
    virus_mat_list = torch.nn.utils.rnn.pack_sequence(list(virus_mat), enforce_sorted=False).to(device).float()
    
    return human_mat_list, virus_mat_list

class HVnet():
    def __init__(self, model_path, thresh, batch_size = 20):
        self.model_path = model_path
        self.batch_size = batch_size
        self.thresh = thresh

    def fit_HVnet(self, data_sets, human_seq_dict, virus_seq_dict):

        data_all = pv_data_sets(data_sets, human_seq_dict, virus_seq_dict)
        loader = DataLoader(dataset = data_all, collate_fn = collate_fn, batch_size = self.batch_size)

        net = unitNET(device).to(device)
        net.load_state_dict(torch.load(self.model_path, map_location = device))

        print("The number of data:" + str(len(data_sets)))

        all_probs, enc_vector_h, enc_vector_v, att_weight_wh, att_weight_wv = [], [], [], [], []
        print("predicting...")
        net.eval()
        for i, (human_mat, virus_mat) in enumerate(loader):
            with torch.no_grad():
                outputs = net(human_mat, virus_mat)
                all_probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                enc_vector_h.extend(net.output_wh.cpu().detach().numpy())
                enc_vector_v.extend(net.output_wv.cpu().detach().numpy())
                att_weight_wh.extend([net.att_h[j].squeeze().cpu().detach().numpy().tolist() for j in range(len(net.att_h))])
                att_weight_wv.extend([net.att_v[j].squeeze().cpu().detach().numpy().tolist() for j in range(len(net.att_v))])
                
        all_probs = [["interact", all_probs[i]] if(all_probs[i] > self.thresh) else ["not interact", all_probs[i]] for i in range(len(all_probs))]

        return pd.DataFrame(all_probs), pd.DataFrame(enc_vector_h), pd.DataFrame(enc_vector_v), att_weight_wh, att_weight_wv
    
def save_results(path, filename, data):
    stringio = io.StringIO()
    writer = csv.writer(stringio)
    writer.writerows(data)
    data = stringio.getvalue().encode("utf-8")
    with open(os.path.join(path, filename), "wb") as f:
        f.write(data)


def pred_main(in_path, out_path, w2v_model_path, deep_model_path, thresh, batch_size, k_mer):
    data_sets, human_seq_dict, virus_seq_dict = make_vectors(in_path, w2v_model_path, num = k_mer)
    deep_net = HVnet(deep_model_path, thresh, batch_size)
    probs, human_vector, viral_vector, human_attention, viral_attention = deep_net.fit_HVnet(data_sets, human_seq_dict, virus_seq_dict)
    
    os.makedirs(out_path, exist_ok = True)
    probs.to_csv(out_path + "/result.csv", sep=",", index = None, header = None)
    human_vector.to_csv(out_path + "/human_transformed_vec.csv", sep=",", index = None, header = None)
    viral_vector.to_csv(out_path + "/viral_transformed_vec.csv", sep=",", index = None, header = None)
    save_results(out_path, "human_protein_attention_weights.csv", human_attention)
    save_results(out_path, "viral_protein_attention_weights.csv", viral_attention)














































































