import json
import sys
import argparse
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F 
from sklearn.preprocessing import MinMaxScaler
from module import Seq2Seq, Encoder, Decoder, Discriminator, Generator, OCAN
'''
@article{zheng2018one,
  title={One-Class Adversarial Nets for Fraud Detection},
  author={Zheng, Panpan and Yuan, Shuhan and Wu, Xintao and Li, Jun and Lu, Aidong},
  journal={arXiv preprint arXiv:1803.01798},
  year={2018}
}
'''

parse = argparse.ArgumentParser(description="""
                    pytorch implementation of OCAN: One Class Adersarial Nets for Fraud Detection
                    -------------------------------------------------------------------
                        @article{zheng2018one,
                        title={One-Class Adversarial Nets for Fraud Detection},
                        author={Zheng, Panpan and Yuan, Shuhan and Wu, Xintao and Li, Jun and Lu, Aidong},
                        journal={arXiv preprint arXiv:1803.01798},
                        year={2018}
                        }
                    --------------------------------------------------------------------
                    \nsee arXiv: https://arxiv.org/pdf/1803.01798.pdf. 
                    \nConfigure you own network in config.json along with you own data""")

parse.add_argument('--data', type=str, default = 'wiki', \
                             help = textwrap.dedent('''
                             Name of the data to be trained. 
                             Official: ('wiki', 'credit_card', 'raw_credit_card').
                             Customized: specify your own name for customized data'''))
parse.add_argument('benign_path', type = str, \
                             help = '(numpy array). Path to the benign data to be trained')
parse.add_argument('vandal_path', type = str,  \
                             help = '(numpy array). Path to the vandal data')
parse.add_argument('--test_x', type = str, default=None,
                             help = '(numpy array). Path to the customized  X test data. \
                             If not provided, test_x will draw from benign_path and vandal_path ')
parse.add_argument('--test_y', type = str, default =None,
                             help = '(numpy array). Path to the customized Y label along with test_x. \
                             `0` for benign, `1` for vandal. If not provided, test_y will be created \
                             according to benign_path and vandal_path')




args = parse.parse_args()

# en_ae = int(sys.argv[1]) # en_ae == 1 for wiki dataset with auto

# dra_tra_pro = int(sys.argv[2])

min_max_scaler = MinMaxScaler()

x_benign = min_max_scaler.fit_transform(np.load(args.benign_path))
x_vandal = min_max_scaler.transform(np.load(args.vandal_path))

# if en_ae == 1:
#     x_benign = min_max_scaler.fit_transform(np.load("./data/hidden/wiki/ben_hid_emd_4_50_8_200_r0.npy"))
#     x_vandal = min_max_scaler.transform(np.load("./data/hidden/wiki/val_hid_emd_4_50_8_200_r0.npy"))
# elif en_ae == 2:
#     x_benign = min_max_scaler.fit_transform(np.load("./data/hidden/credit_card/ben_hid_repre_r2.npy"))
#     x_vandal = min_max_scaler.transform(np.load("./data/hidden/credit_card/van_hid_repre_r2.npy"))
# else:
#     x_benign = min_max_scaler.fit_transform(np.load("./data/hidden/raw_credit_card/ben_raw_r0.npy"))
#     x_vandal = min_max_scaler.transform(np.load("./data/hidden/raw_credit_card/van_raw_r0.npy"))


x_benign = sample_shuffle_uspv(x_benign)
x_vandal = sample_shuffle_uspv(x_vandal)

if args.data == 'credit_card' or args.data == 'raw_credit_card':
    x_pre = x_benign[0:700]
else:
    x_benign = x_benign[0:10000]
    x_vandal = x_vandal[0:10000]
    x_pre = x_benign[0:7000]

y_pre = np.zeros(len(x_pre))
y_pre = one_hot(y_pre, 2)

x_train = x_pre

if args.test_x:
    if not args.test_y:
        raise ValueError("You must provide y if you provide x")
    else:
        x_test =min_max_scaler.transform(np.load(args.test_x))
        y_test = np.load(args.test_y)
elif args.data == 'wiki':
    x_test = np.vstack([x_benign[-3000:], x_vandal[-3000:]])
    y_test = np.zeros(x_test.shape[0])
    y_test[3000:]=1
else:
    x_test = np.vstack([x_benign[-490:],x_vandal[-490:]])
    y_test = np.zeros(x_test.shape[0])
    y_test[490:]=1


cfg = json.load(open('./config.json'))

net_cfg, train_cfg = cfg['net_cfg'], cfg['train_cfg']
# print(train_cfg)
ocan_model = OCAN(net_cfg, train_cfg)

# print(ocan_model.seq2seq)
print(ocan_model.generator)
print(ocan_model.discriminator)
print(ocan_model.discriminator_r)

print("\n start regular GAN pretraining")
ocan_model.pretrain_rGAN(torch.Tensor(x_benign), **train_cfg['rGAN'])


print("\n start complementary GAN training")
result = ocan_model.train_cGAN(torch.Tensor(x_benign),
                      torch.Tensor(x_vandal),
                      torch.Tensor(x_test),
                      y_test,
                      **train_cfg['cGAN'])

draw_trend(result['benign_prob_train'],
           result['gen_prob_train'],
           result['van_prob_train'],
           result['fm_loss'],
           result['pt_loss'],
           result['trld_loss'],
           result['p_s'],
           result['r_s'],
           result['f1_s'],
           result['auc_s'])
exit(0)