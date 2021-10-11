

import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("./")

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch_optimizer as optim
from torch.utils.data import BatchSampler
import numpy as np
import argparse
from gensim.models import word2vec
from deep_net import unitNET
from loss_func import CBLoss
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, AUPRC
from sklearn.model_selection import StratifiedKFold
metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def file_input_csv(filename,index_col = None):
    data = pd.read_csv(filename, index_col = index_col)

    return data

def create_vector(data, index, model, num = 4):
    seq_dict = {}
    id_dict = {}

    seq = list(set(data[index].values.tolist()))

    for i in range(len(seq)):
        seq_dict[i] = [model[seq[i][j:j + num]]  for j in range(len(seq[i]) - num + 1)]
        id_dict[seq[i]] = i

    return seq_dict, id_dict

def change_seq_to_id(data, id_dict, index):
    for i in range(len(data.reset_index(drop=True))):
        data.loc[i, index] = id_dict[data.loc[i, index]]

    return data

def make_vectors(train_data_sets, val_data_sets, wv_model, num = 4):

    data_all = pd.concat([train_data_sets, val_data_sets])
    human_seq_dict, human_id_dict = create_vector(data_all, "human_seq", wv_model, num = num)
    virus_seq_dict, virus_id_dict = create_vector(data_all, "virus_seq", wv_model, num = num)

    train_data_sets = change_seq_to_id(train_data_sets, human_id_dict, "human_seq")
    train_data_sets = change_seq_to_id(train_data_sets, virus_id_dict, "virus_seq")
    val_data_sets = change_seq_to_id(val_data_sets, human_id_dict, "human_seq")
    val_data_sets = change_seq_to_id(val_data_sets, virus_id_dict, "virus_seq")

    return train_data_sets, val_data_sets, human_seq_dict, virus_seq_dict


class pv_data_sets(data.Dataset):
    def __init__(self,data_sets, human_seq_dict, virus_seq_dict):
        super().__init__()
        self.human_seq = data_sets["human_seq"].values.tolist()
        self.virus_seq = data_sets["virus_seq"].values.tolist()
        self.labels = np.array(data_sets["labels"].values.tolist()).reshape([len(data_sets["labels"].values.tolist()),1]).astype(np.float32)
        self.human_seq_dict = human_seq_dict
        self.virus_seq_dict = virus_seq_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        human_mat = self.human_seq_dict[self.human_seq[idx]]
        virus_mat = self.virus_seq_dict[self.virus_seq[idx]]
        labels = self.labels[idx]
        
        return torch.tensor(human_mat), torch.tensor(virus_mat), labels

def collate_fn(batch):
    human_mat, virus_mat, labels = list(zip(*batch))

    human_mat_list = torch.nn.utils.rnn.pack_sequence(list(human_mat), enforce_sorted=False).to(device).float()
    virus_mat_list = torch.nn.utils.rnn.pack_sequence(list(virus_mat), enforce_sorted=False).to(device).float()

    return human_mat_list, virus_mat_list, torch.tensor(labels).to(device).float()

class BinarySampler(BatchSampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.index = np.arange(len(self.labels))
        np.random.shuffle(self.index)
        self.labels_temp = self.labels[self.index]
        self.batch_count = int(len(self.labels)/batch_size)
        self.index = self.index[0:self.batch_count*batch_size]
        self.labels_temp = self.labels_temp[0:self.batch_count*batch_size]
       
        self.batch_index_all = []
        self.skf = StratifiedKFold(self.batch_count)
        for _, batch_index in self.skf.split(self.index, self.labels_temp):
            self.batch_index_all.append(batch_index)

    def __iter__(self):
        for i in range(len(self.batch_index_all)):
            yield self.index[self.batch_index_all[i]]

    def __len__(self):
        return self.batch_count
    
   
def print_out(comment, f):
    print(comment, file = f, flush=True)
    print(comment, flush=True)

class HVnet():
    def __init__(self, out_path, tra_batch_size = 1024, val_batch_size = 1024, features = 128, lr = 0.001, n_epoch = 10000, early_stop = 20, thresh = 0.5, loss_type = "imbalanced"):
        self.out_path = out_path
        self.tra_batch_size = tra_batch_size
        self.val_batch_size = val_batch_size
        self.features = features
        self.lr = lr
        self.n_epoch = n_epoch
        self.early_stop = early_stop
        self.thresh = thresh
        self.loss_type = loss_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit_HVnet(self, train_data_sets, val_data_sets, human_seq_dict, virus_seq_dict):
        os.makedirs(self.out_path + "/model", exist_ok=True)
        tra_data_all = pv_data_sets(train_data_sets, human_seq_dict, virus_seq_dict)
        balanced_sampler_train = BinarySampler(train_data_sets["labels"].values.tolist(), self.tra_batch_size)
        train_loader = DataLoader(dataset = tra_data_all, collate_fn = collate_fn, batch_sampler = balanced_sampler_train)
   
        val_data_all = pv_data_sets(val_data_sets, human_seq_dict, virus_seq_dict)
        val_loader = DataLoader(dataset = val_data_all, collate_fn = collate_fn, batch_size = self.val_batch_size, shuffle=True)

        net = unitNET(self.device).to(device)
        opt = optim.RAdam(params = net.parameters(), lr = self.lr)
        
        if(self.loss_type == "balanced"):
            criterion = nn.BCELoss()

        max_acc = 0
        early_stop_count = 0

        with open(self.out_path + "/deep_HV_result.txt", 'w') as f:
            print_out(self.out_path, f)
            print_out("The number of training data:" + str(len(train_data_sets)), f)
            print_out("The number of validation data:" + str(len(val_data_sets)), f)
            
            #print(self.out_path, file = f, flush=True)
            #print("The number of training data:" + str(len(train_data_sets)), file = f, flush=True)
            #print("The number of validation data:" + str(len(val_data_sets)), file = f, flush=True)

            for epoch in range(self.n_epoch):
                train_losses, val_losses, train_probs, val_probs, train_labels, val_labels = [], [], [], [], [], []

                print_out("epoch_" + str(epoch + 1) + "=====================", f)
                print_out("train...", f)
                #print("epoch_" + str(epoch + 1) + "=====================", file = f, flush=True) 
                #print("train...", file = f, flush=True)
                
                net.train()
                for i, (human_mat, virus_mat, label) in enumerate(train_loader):
                    opt.zero_grad()
                    outputs = net(human_mat, virus_mat)

                    if(self.loss_type == "balanced"):
                        loss = criterion(outputs, label)
                    elif(self.loss_type == "imbalanced"):
                        loss = CBLoss(label, outputs, 0.99, 2)
                    else:
                        print_out("ERROR::You can not specify the loss type.", f)
                        #print("ERROR::You can not specify the loss type.")  
                        sys.exit()
                    
                    loss.backward()
                    opt.step()

                    train_losses.append(float(loss.item()))
                    train_probs.extend(outputs.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())
                    train_labels.extend(label.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())

                print_out("train_loss:: value: %f, epoch: %d" % (sum(train_losses) / len(train_losses), epoch + 1), f)
                #print("train_loss:: value: %f, epoch: %d" % (sum(train_losses) / len(train_losses), epoch + 1), file = f, flush=True)
                for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](train_labels, train_probs, thresh = self.thresh)
                    else:
                        metrics = metrics_dict[key](train_labels, train_probs)
                    print_out("train_" + key + ": " + str(metrics), f)
                    #print("train_" + key + ": " + str(metrics), file = f, flush=True)

                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(train_labels, train_probs, thresh = self.thresh)
                print_out("train_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), f)
                print_out("train_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), f)
                print_out("train_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), f)
                print_out("train_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), f)
                
                #print("train_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file = f, flush=True)
                #print("train_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file = f, flush=True)
                #print("train_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file = f, flush=True)
                #print("train_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file = f, flush=True)

                print_out("validation...", f)
                #print("validation...", file = f, flush=True)
                net.eval()
                for i, (human_mat, virus_mat, label) in enumerate(val_loader):
                    with torch.no_grad():
                        outputs = net(human_mat, virus_mat)

                        if(self.loss_type == "balanced"):
                            loss = criterion(outputs, label)
                        elif(self.loss_type == "imbalanced"):
                            loss = CBLoss(label, outputs, 0.99, 2)
                        else:
                            print_out("ERROR::You can not specify the loss type.", f)
                            #print("ERROR::You can not specify the loss type.")
                            sys.exit()

                        if(np.isnan(loss.item()) == False):
                            val_losses.append(float(loss.item()))

                        val_probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                        val_labels.extend(label.cpu().detach().squeeze(1).numpy().flatten().tolist()) 
                 
                early_acc = metrics_dict["accuracy"](val_labels, val_probs, thresh = self.thresh)
                loss_epoch = sum(val_losses) / len(val_losses)
                
                print_out("validation_loss:: value: %f, epoch: %d" % (loss_epoch, epoch + 1), f)
                #print("validation_loss:: value: %f, epoch: %d" % (loss_epoch, epoch + 1), file = f, flush=True)
                for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](val_labels, val_probs, thresh = self.thresh)
                    else:
                        metrics = metrics_dict[key](val_labels, val_probs)
                    print_out("validation_" + key + ": " + str(metrics), f)
                    #print("validation_" + key + ": " + str(metrics), file = f, flush=True)

                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(val_labels, val_probs, thresh = self.thresh)
                print_out("validation_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), f)
                print_out("validation_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), f)
                print_out("validation_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), f)
                print_out("validation_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), f)
                
                #print("validation_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file = f, flush=True)
                #print("validation_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file = f, flush=True)
                #print("validation_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file = f, flush=True)
                #print("validation_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file = f, flush=True)

                if early_acc > max_acc:
                    early_stop_count = 0
                    max_acc = early_acc
                    os.makedirs(self.out_path + "/model", exist_ok=True)
                    os.chdir(self.out_path + "/model")
                    torch.save(net.state_dict(), "deep_model")
                    final_val_probs = val_probs
                    final_val_labels = val_labels
                    final_train_probs = train_probs
                    final_train_labels = train_labels
                else:
                    early_stop_count += 1
                    if early_stop_count >= self.early_stop:
                        print_out('Traning can not improve from epoch {}\tBest acc: {}'.format(epoch + 1 - self.early_stop, max_acc), f)
                        #print('Traning can not improve from epoch {}\tBest acc: {}'.format(epoch + 1 - self.early_stop, max_acc), file = f, flush=True)
                        break
                    
                print_out("", f)
                #print("", file = f, flush=True)
                
            print_out("", f)
            #print("", file = f, flush=True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    train_metrics = metrics_dict[key](final_train_labels,final_train_probs,thresh = self.thresh)
                    val_metrics = metrics_dict[key](final_val_labels,final_val_probs, thresh = self.thresh)
                else:
                    train_metrics = metrics_dict[key](final_train_labels, final_train_probs)
                    val_metrics = metrics_dict[key](final_val_labels, final_val_probs)
                print_out("train_" + key + ": " + str(train_metrics), f)
                print_out("test_" + key + ": " + str(val_metrics), f)
                #print("train_" + key + ": " + str(train_metrics), file = f, flush=True)
                #print("test_" + key + ": " + str(val_metrics), file = f, flush=True)


def training_main(training_data_path, validation_data_path, word2vec_model_path, outpath, losstype, training_batch_size, validation_batch_size, learning_rate, max_epoch_num, early_stopping_epoch_num, threshold, k_num):
    if(0 > threshold and threshold > 1):
        print("ValueError: threshold must be a value from 0 to 1")

    train_data_sets = file_input_csv(training_data_path, index_col = None)
    val_data_sets = file_input_csv(validation_data_path, index_col = None)
    wv_model = word2vec.Word2Vec.load(word2vec_model_path)

    print("preparing")
    train_data_sets, val_data_sets, human_seq_dict, virus_seq_dict = make_vectors(train_data_sets, val_data_sets, wv_model, num = k_num)

    print("training")
    deep_net = HVnet(out_path=outpath, tra_batch_size = training_batch_size, val_batch_size = validation_batch_size, features = wv_model.vector_size, lr = learning_rate, n_epoch = max_epoch_num, early_stop = early_stopping_epoch_num, thresh = threshold, loss_type = losstype) 
    deep_net.fit_HVnet(train_data_sets, val_data_sets, human_seq_dict, virus_seq_dict)



"""
parser = argparse.ArgumentParser()
parser.add_argument('-tp', '--training_data_path', help='Path of training data', required=True)
parser.add_argument('-vp', '--validation_data_path', help='Path of validation data', required=True)
parser.add_argument('-wm', '--word2vec_model_path', help='Path of a trained word2vec model', required=True)
parser.add_argument('-o', '--outpath', help='Directory to output results', required=True)
parser.add_argument('-l', '--losstype', help='Loss type (imbalanced: loss function for imbalanced data, balanced: Loss function for balanced data)', default = "imbalanced", choices=["balanced", "imbalanced"])
parser.add_argument('-t_batch', '--training_batch_size', help='Training batch size', default = 1024, type=int)
parser.add_argument('-v_batch', '--validation_batch_size', help='Validation batch size', default = 1024, type=int)
parser.add_argument('-lr', '--learning_rate', help='learning rate', default = 0.001, type=float)
parser.add_argument('-max_epoch', '--max_epoch_num', help='maximum epoch number', default = 10000, type=int)
parser.add_argument('-stop_epoch', '--early_stopping_epoch_num', help='epoch number for early stopping', default = 20, type=int)
parser.add_argument('-thr', '--threshold', help='threshold to determined whether interact or not', default = 0.5, type=float)
parser.add_argument('-k_mer', '--number_of_k', help='size of k in k_mer', default = 4, type=int)

training_data_path = parser.parse_args().training_data_path
validation_data_path = parser.parse_args().validation_data_path
word2vec_model_path = parser.parse_args().word2vec_model_path
outpath = parser.parse_args().outpath
losstype = parser.parse_args().losstype
training_batch_size = parser.parse_args().training_batch_size
validation_batch_size = parser.parse_args().validation_batch_size
learning_rate = parser.parse_args().learning_rate
max_epoch_num = parser.parse_args().max_epoch_num
early_stopping_epoch_num = parser.parse_args().early_stopping_epoch_num
threshold = parser.parse_args().threshold
k_num = parser.parse_args().number_of_k
"""












































































