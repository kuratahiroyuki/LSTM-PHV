
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_model_summary import summary
import torch.nn.utils.rnn as rnn

class unitNET(nn.Module):
    def __init__(self, device, features = 128, lstm_hidden_size = 64):
        super(unitNET, self).__init__()
        self.features = features
        self.lstm_hidden_size = lstm_hidden_size
        self.device = device
        
        self.lstm_human = nn.LSTM(self.features, self.lstm_hidden_size, batch_first=True, bidirectional=True)
        self.lstm_virus = nn.LSTM(self.features, self.lstm_hidden_size, batch_first=True, bidirectional=True)
        
        self.dense_1_human = nn.Linear(self.lstm_hidden_size*2, 64)
        self.dense_2_human = nn.Linear(64, 32)
        self.dense_3_human = nn.Linear(32, 1)
        
        self.dense_1_virus = nn.Linear(self.lstm_hidden_size*2, 64)
        self.dense_2_virus = nn.Linear(64, 32)
        self.dense_3_virus = nn.Linear(32, 1)
        
        self.dense_both_1 = nn.Linear(self.lstm_hidden_size*2*2, 200)
        self.dense_both_2 = nn.Linear(200, 100)
        self.dense_both_3 = nn.Linear(100, 40)
        self.dense_both_4 = nn.Linear(40, 1)
        
        self.broad_one = torch.ones(1, self.lstm_hidden_size*2).to(self.device)
        self.dense_dropout = nn.Dropout(p = 0.3)
        self.lstm_dropout = nn.Dropout(p = 0.3)
        
    def time_distributed_dropout(self, output_h, output_v, length_h, length_v):
        output_h_temp = [self.lstm_dropout(output_h[i,0:length_h[i],:]) for i in range(output_h.size()[0])]
        output_v_temp = [self.lstm_dropout(output_v[i,0:length_v[i],:]) for i in range(output_v.size()[0])]

        output_h = torch.nn.utils.rnn.pad_sequence(output_h_temp, batch_first = True)
        output_v = torch.nn.utils.rnn.pad_sequence(output_v_temp, batch_first = True)

        return output_h, output_v
        

    def forward(self, human_we_mat, virus_we_mat):
        output_h, _ = self.lstm_human(human_we_mat)
        hidden_state_h, length_h = torch.nn.utils.rnn.pad_packed_sequence(output_h, batch_first =True)

        output_v, _ = self.lstm_virus(virus_we_mat)
        hidden_state_v, length_v = torch.nn.utils.rnn.pad_packed_sequence(output_v, batch_first =True)
    
        output_h = F.relu(self.dense_1_human(hidden_state_h))
        output_v = F.relu(self.dense_1_virus(hidden_state_v))

        output_h, output_v = self.time_distributed_dropout(output_h, output_v, length_h, length_v)

        output_h = F.relu(self.dense_2_human(output_h))
        output_v = F.relu(self.dense_2_virus(output_v))

        output_h, output_v = self.time_distributed_dropout(output_h, output_v, length_h, length_v)

        output_h = self.dense_3_human(output_h)
        output_v = self.dense_3_virus(output_v)

        self.att_h = []
        self.att_v = []
        for i in range(output_h.size()[0]):
            self.att_h.append(F.softmax(output_h[i,0:length_h[i],:] ,dim=0))
            self.att_v.append(F.softmax(output_v[i,0:length_v[i],:] ,dim=0))
       
        att_h = rnn.pad_sequence(self.att_h, batch_first=True)
        att_v = rnn.pad_sequence(self.att_v, batch_first=True)
 
        output_h = torch.matmul(att_h,self.broad_one)
        output_v = torch.matmul(att_v,self.broad_one)
 
        output_h = hidden_state_h*output_h
        output_v = hidden_state_v*output_v
        
        self.output_wh = output_h.sum(dim=1)
        self.output_wv = output_v.sum(dim=1)
        
        output = torch.cat([self.output_wh,self.output_wv],dim=1)
       
        output = F.relu(self.dense_both_1(output))
        output = self.dense_dropout(output)
       
        output = F.relu(self.dense_both_2(output))
        output = self.dense_dropout(output)
        
        output = F.relu(self.dense_both_3(output))
        output = self.dense_dropout(output)
 
        output = F.sigmoid(self.dense_both_4(output)) 
        
        return output




















