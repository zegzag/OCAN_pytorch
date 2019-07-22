import numpy as np
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import \
        precision_score, f1_score, recall_score, roc_auc_score
from scipy.stats import ks_2samp
from utils import *

#AutoEncoder Module
class Encoder(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers=1):
        super().__init__()
        self.rnn=nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
    def forward(self,input_seq): #input_seq: (batch, seq_len, feature_size)
        _, (hn, cn)=self.rnn(input_seq)
        return hn
    
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, seq_len, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(hidden_size, output_size,num_layers, batch_first = True)
        self.seq_len = seq_len
    def forward(self, last_h): #last_hidden: (1, batch, hidden_size)
        sum_seq=last_h.expand(self.seq_len,-1,-1) # (seq_len, batch, hidden_size)
        sum_seq = torch.transpose(sum_seq, 0, 1) # (batch, seq_len, hidden_size)
        output, _ = self.rnn(sum_seq)
        return output

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, num_layers=1):
        super().__init__()
        self.encoder=Encoder(input_size, hidden_size, num_layers)
        self.decoder=Decoder(hidden_size, input_size, seq_len, num_layers)
    def forward(self,input_seq):
        hn=self.encoder(input_seq)
        output = self.decoder(hn)
        return output, hn

#Dense Module of GAN
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, is_gen=False):
        super().__init__()
        dim_list=[]
        dim_list.insert(0, input_dim)
        dim_list.extend(hidden_dims)
        dim_list.append(output_dim)
        
        self.Layers=nn.ModuleList()
        for i in range(len(dim_list)-1):
            self.Layers.append(nn.Linear(dim_list[i], dim_list[i+1]))
            self.Layers.append(nn.ReLU())
        if is_gen:
            self.Layers[-1]=nn.Tanh()
        else:
            self.Layers[-1]=nn.Softmax(dim=1)
    def forward(self, true_data):
        h_out=[]
        h_out.append(true_data)
        for layer in self.Layers:
            h = layer(h_out[-1])
            h_out.append(h)
        return h_out[-1],h_out[-2],h_out[-3]

class Generator(Discriminator):
    def __init__(self, input_dim, hidden_dims, output_dim, is_gen=True):
        super().__init__(input_dim, hidden_dims, output_dim, is_gen)
    def forward(self, fake_data):
        prob, logits, h=super(Generator,self).forward(fake_data)
        return prob, logits, h


#OCAN Algorithm implementation
class OCAN():
    def __init__(self,net_cfg, train_cfg):

        self.seq2seq=Seq2Seq(**net_cfg['seq2seq_cfg'])
        self.generator=Generator(**net_cfg['g_cfg'])
        self.discriminator=Discriminator(**net_cfg['d_cfg'])
        self.discriminator_r=Discriminator(**net_cfg['d_cfg'])
        
        ## xavier_normal initialization 
        for i, param in enumerate(self.generator.parameters()):
            try:
                nn.init.xavier_normal_(param.data)
            except:
                pass
        for i, param in enumerate(self.discriminator.parameters()):
            try:
                nn.init.xavier_normal_(param.data)
            except:
                pass
        for i, param in enumerate(self.discriminator_r.parameters()):
            try:
                nn.init.xavier_normal_(param.data)
            except:
                pass
    
    def train_autoencoder(self, dataset, BATCH_SIZE, EPOCH, LR, is_hidden = False):
        r"""Train autoencoder using LSTM cell.

        Shape:
            - Input: :math: (seq_len, batch, input_size): tensor containing the fatures
                of the input sequence
        """
        loader = DataLoader(TensorDataset(dataset), batch_size=BATCH_SIZE,shuffle=True)
        optimizer=torch.optim.Adam(self.seq2seq.parameters(),lr=LR)
        for epoch in range(EPOCH):
            for i, benign in enumerate(loader):
                bt=time.time()
                optimizer.zero_grad()
                benign_ft = benign[0]

                benign_gen, _=self.seq2seq(benign_ft)
                loss=F.mse_loss(benign_gen, benign_ft)
                loss.backward()
                optimizer.step()
                et=time.time()
                print('Epoch: {} | Batch: {} | Loss: {} | Durr: {}'.format(epoch, i, loss, et-bt))
        
        return self.seq2seq
    def pretrain_rGAN(self, benign_train, BATCH_SIZE, EPOCH, LR):
        r"""pre-train net for density estimation
        Shape:
            benign_train (N, h_dim): Tensor. the last hidden state of encoder of AutoEncoder model under benign data.
        """
        loader = DataLoader(TensorDataset(benign_train), batch_size = BATCH_SIZE, shuffle = True)
        d_optimizer_r = torch.optim.SGD(self.discriminator_r.parameters(), lr=LR)
        y_pre = torch.ones(BATCH_SIZE).to(torch.long) 
        
        for epoch in range(EPOCH):
            for i, benign in enumerate(loader):

                bt = time.time()
                benign_ft = benign[0]
                
                d_optimizer_r.zero_grad()
                prob, logits, h2 = self.discriminator_r(benign_ft)
                y_pre = torch.zeros(benign_ft.shape[0]).to(torch.long)

                loss =nn.CrossEntropyLoss()(logits, y_pre)
                loss.backward()
                d_optimizer_r.step()
                et = time.time()
            print('Epoch: {} | Loss: {} | Durr: {}'.format(epoch, loss, et-bt))
                
    def train_cGAN(self, benign_train, van_train, test_df, y_test, BATCH_SIZE,EPOCH, LR, g_dim, alpha):
        r"""complementory GAN Training
        Args:
            g_dim (int): input dim of generator
            alpha (float): weight of entropy loss of `loss_dsc_c`
        Shape:
            benign_train (N, h_dim): Tensor. Last hidden state of Encoder of benign data
            van_train (N, h_dim): Tensor. Last hidden state of Encoder of vandal data
            test_df (N, h_dim): Tensor. Last hidden state of test data (benign + vandal)
            y_test (N,): numpy. y label of `test_df`
        """
        loader = DataLoader(TensorDataset(benign_train), batch_size=BATCH_SIZE, shuffle=True)
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=LR)
        d_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=LR)


        fm_loss = []
        pt_loss = []
        trld_loss = []
        
        ## prob trend track
        benign_prob_train = []
        gen_prob_train = []
        van_prob_train = []
        
        ## model evaluation
        p_s = []
        recall_s = []
        f1_s = []
        auc_s =[]
        ks_v = []

        for epoch in range(EPOCH):
            bt=time.time()
            for i, benign in enumerate(loader):
                
                benign_ft = benign[0]
                g_optimizer.zero_grad()
                d_optimizer.zero_grad()
                
                y_gen=torch.ones(benign_ft.shape[0]).to(torch.long)
                y_data=torch.zeros(benign_ft.shape[0]).to(torch.long)

                sample_ft, _, _ = self.generator(sampleZ(benign_ft.shape[0], g_dim))
    
                prob_gen_c, logits_gen_c, h_gen_c = self.discriminator(sample_ft)
                prob_data_c, logits_data_c, h_data_r= self.discriminator(benign_ft)
                prob_gen_r, logits_gen_r, h_gen_r = self.discriminator_r(sample_ft)
                
                loss_d=loss_dsc_c(logits_gen_c, logits_data_c, y_gen, y_data, alpha)
                loss_g, loss_pt, loss_fm, loss_trld=loss_gen_c(h_gen_r, prob_gen_r, logits_gen_c, logits_data_c)
                
                
                loss_d.backward(retain_graph=True)
                d_optimizer.step()
                
                loss_g.backward()
                g_optimizer.step()

            et = time.time()
            
            with torch.no_grad():
                gen_train, _, _ = self.generator(sampleZ(benign_train.shape[0], g_dim))

                probs_benign_train, _, _ = self.discriminator(benign_train)
                probs_gen_train, _, _ = self.discriminator(gen_train)
                probs_van_train, _, _ = self.discriminator(van_train)

                prob_benign_train = np.mean(probs_benign_train[:,0].detach().numpy())
                prob_gen_train = np.mean(probs_gen_train[:,0].detach().numpy())
                prob_van_train = np.mean(probs_van_train[:,0].detach().numpy())

                benign_prob_train.append(prob_benign_train)
                gen_prob_train.append(prob_gen_train)
                van_prob_train.append(prob_van_train)

                loss_fm = float(loss_fm.detach().numpy())
                loss_pt = float(loss_pt.detach().numpy())
                loss_trld = float(loss_trld.detach().numpy())

                fm_loss.append(loss_fm)
                pt_loss.append(loss_pt)
                trld_loss.append(loss_trld)
            
            prob_test, _, _ = self.discriminator(test_df)
            prob_test = prob_test.detach().numpy()
            label_test = np.argmax(prob_test, axis=1)

            # import pdb; pdb.set_trace()
            ps = precision_score(y_test, label_test)
            rs = recall_score(y_test, label_test)
            f1 = f1_score(y_test, label_test)
            # import pdb
            # pdb.set_trace()
            auc = roc_auc_score(y_test, prob_test[:,1])

            # import pdb
            # pdb.set_trace()
            ksv, _ = ks_2samp(prob_test[label_test==0,0], prob_test[label_test==1,0])

            print("Epoch: {} | DiscLoss: {:0.4f} | GenLoss: {:0.4f} | Precision: {:0.4f}| Recall: {:0.4f} | F1: {:0.4f} | AUC: {:0.4f} | KS: {:0.4f} Durr: {:0.4f}" \
                 .format(epoch, loss_d, loss_g, ps, rs, f1, auc, ksv, et-bt))

            p_s.append(ps)
            recall_s.append(rs)
            f1_s.append(f1)
            auc_s.append(auc)

        result ={
            'fm_loss':fm_loss,
            'pt_loss':pt_loss,
            'trld_loss':trld_loss,
            'benign_prob_train':benign_prob_train,
            'gen_prob_train': gen_prob_train,
            'van_prob_train':van_prob_train,
            
            'p_s':p_s,
            'r_s':recall_s,
            'f1_s':f1_s,
            'auc_s':auc_s,
            'ks_v':ks_v

        }
        return result