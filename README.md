 # OCAN： One-Class Adversarial Nets fo Fraud Detection [pytorch Implementation]
 Reference:  
 [official implementation](https://github.com/PanpanZheng/OCAN)  
 [paper link](https://arxiv.org/pdf/1803.01798.pdf)

 # Running Environment
 python 3.7.1  
 pytorch 1.0.1

 # Guideline
 + For help: `python main.py -h`
 + For wiki data:

    1.modify `net_cfg['g_cfg']['output_dim']=200; net_cfg['g_cfg']['d_cfg']['input_dim']=200;`in _config.json_ 

    2.run `python main.py ./data/hidden/wiki/ben_hid_emd_4_50_8_200_r0.npy ./data/hidden/wiki/val_hid_emd_4_50_8_200_r0.npy`
 + For credit data: just modify the same configuration consistent with the data dimension. `credit_card: 50, raw_credit_card: 30`

 + For your own data: You can customize the net structure and training hyperparameters of OCAN in _config.json_

 

## Welcome for bugs and issues reporting
