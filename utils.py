import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 

#Loss function defination
def pt_loss(h_gen_r):
    r"""pull-away(PT) term for approximation to negtive `pG` entropy
    Args:
        h_gen_r (batch, hidden_dim): tensor of the hidden layer of regular GAN
    Return:
        ptloss (float): PT Loss
    """
    N=h_gen_r.shape[0]
    pt_idx=1-torch.eye(N,N)
    pt_idx=pt_idx.to(torch.uint8)
    a=torch.matmul(h_gen_r, h_gen_r.transpose(1,0))
    h_norm=torch.norm(h_gen_r, dim=1)
    h_norm = torch.where(h_norm!=0, h_norm, torch.ones_like(h_norm))
    h_norm = h_norm.reshape(-1,1)
    b=torch.matmul(h_norm, h_norm.transpose(1,0))
    c=a/b
    c=torch.sum(c[pt_idx])
    z=N*(N-1)
    ptloss= c/z
    return ptloss
    
    
def loss_gen_c(h_gen_r, prob_gen_r, logits_gen_c, logits_data_c):
    r"""loss of the generation. [Paper Eq. (14)]
    Args:
        h_gen_r (batch, hidden_dim): tensor of the hidden layer of Regular Discriminator.
        prob_gen_r (batch, 2): prob (after softmax) output of the Regular Discriminator on Generation Z
        logits_gen_c (batch, 2): logits output (before softmax)  of y the Complementary Discriminator on Generation Z
        logits_data_c (batch, 2): logits output (before softmax) of the Complementory  Discriminator on True Data
    Return:
        loss (float): loss function for generator backward.
    """
    ptloss=pt_loss(h_gen_r)
    diff = logits_data_c-logits_gen_c
    fm_loss = torch.mean(
                torch.sqrt(
                    torch.sum(diff*diff, dim = 1)
                          )
                        )

    e=(torch.max(prob_gen_r[:,-1])+torch.min(prob_gen_r[:,-1]))/2    

    mask_tar =torch.where(prob_gen_r[:,-1]>e, 
                          torch.ones_like(prob_gen_r[:,-1]),torch.zeros_like(prob_gen_r[:,-1]))
    thrld_loss = torch.mean(torch.log(prob_gen_r[:,-1])*mask_tar)
    
    loss= ptloss + fm_loss + thrld_loss
    return loss, ptloss, fm_loss, thrld_loss
    
def loss_dsc_c(logits_gen_c, logits_data_c, y_gen, y_data, alpha):
    r"""loss of the complementary discriminator 
    Args:
        logits_gen_c (batch, 2): logits output (before softmax)  of y the Complementary Discriminator on Generation Z
        logits_data_c (batch, 2): logits output (before softmax) of the Complementory  Discriminator on True Data
        y_gen (batch, ): y label of generation Z. `1` assigned here
        y_data (batch, ): y label of true data. `0` assigned here
    """

    loss1=nn.CrossEntropyLoss()(logits_data_c, y_data)
    loss2=nn.CrossEntropyLoss()(logits_gen_c, y_gen)

    loss3= -torch.mean( \
                torch.sum( \
                    F.softmax(logits_data_c, dim=1)*F.log_softmax(logits_data_c, dim = 1),dim = 1
                ))
        
    loss = loss1 + loss2 + alpha*loss3
    return loss

#Other function
def sampleZ(batch_size, g_dim):
    noise = np.random.uniform(-1., 1., size = (batch_size, g_dim))
    return torch.Tensor(noise)

def shuffle_spv(X, y):
    num_df = X.shape[0]
    s = np.arange(num_df)
    np.random.shuffle(s)
    return X[s], y[s]

def sample_shuffle_uspv(X):
    n_samples = len(X)
    s = np.arange(n_samples)
    np.random.shuffle(s)
    return np.array(X[s])

def one_hot(x, depth):
    x_one_hot = np.zeros((len(x), depth), dtype=np.int32)
    x = x.astype(int)
    for i in range(x_one_hot.shape[0]):
        x_one_hot[i, x[i]] = 1
    return x_one_hot


def draw_trend(D_real_prob, D_fake_prob, D_val_prob, \
               fm_loss, pt_loss, thrld_loss, \
               p_s, r_s, f1, auc):

    fig = plt.figure()
    fig.patch.set_facecolor('w')
    # plt.subplot(311)
    p1, = plt.plot(D_real_prob, "-g")
    p2, = plt.plot(D_fake_prob, "--r")
    p3, = plt.plot(D_val_prob, ":c")
    plt.xlabel("# of epoch")
    plt.ylabel("probability")
    leg = plt.legend([p1, p2, p3], [r'$p(y|V_B)$', r'$p(y|\~{V})$', r'$p(y|V_M)$'], loc=1, bbox_to_anchor=(1, 1), borderaxespad=0.)
    leg.draw_frame(False)
    # plt.legend(frameon=False)
    
    
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    # plt.subplot(312)
    p41, = plt.plot(fm_loss, "-b")
    p42, = plt.plot(pt_loss, "-r")
    p43, = plt.plot(thrld_loss, "-g")
    plt.xlabel("# of epoch")
    plt.ylabel("loss")
    plt.legend([p41, p42, p43], ['fm_loss','pt_loss', 'thrld_loss'], loc=1, bbox_to_anchor=(1, 1), borderaxespad=0.)

    fig = plt.figure()
    fig.patch.set_facecolor('w')
    # plt.subplot(313)
    p5, = plt.plot(p_s, "-g")
    p6, = plt.plot(r_s, "-r")
    p7, = plt.plot(f1, "-b")
    p8, = plt.plot(auc, "-y")
    leg = plt.legend([p5, p6, p7, p8], ['precision', 'recall', 'f1', 'auc'], loc = 1, bbox_to_anchor=(1,1),borderaxespad = 0.)
    plt.xlabel("# of epoch")
    plt.ylabel("metrics")
    
    

    # plt.legend([p1, p2, p3, p4, p5], ["d_real_prob", "d_fake_prob", "d_val_prob", "fm_loss","f1"], loc=1, bbox_to_anchor=(1, 3.5), borderaxespad=0.)
    plt.show()