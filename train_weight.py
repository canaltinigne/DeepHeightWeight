import numpy as np 
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

from loss import dice_coef, dice_loss
from dataset_weight import Images
from network_weight import UNet
from datetime import datetime
from glob import glob

"""
Training Weight Estimation Network

Example Run:
CUDA_VISIBLE_DEVICES=1 python train_weight.py -e 50 -g 0 -l 1e-3 -ls ce -d [DIRECTORY OF TRAINING FILE] -opt adam -n 100 -bs 8

"""

if __name__ == "__main__":
    
    # PARSER SETTINGS
    parser = argparse.ArgumentParser(description='U-Net PyTorch Model for Height and Weight Prediction in IMDB Dataset')

    parser.add_argument('-e', '--epoch', type=int, required=True, help='Number of Epochs')
    parser.add_argument('-l', '--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('-ls', '--loss', type=str, required=True, help='Height loss type')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Dataset directory')
    
    parser.add_argument('-m', '--min_neuron', type=int, default=128, help='Minimum neuron number for the first layer')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-opt', '--optimizer', type=str, required=True, help='Optimizer')
    parser.add_argument('-n', '--normalize', type=float, required=True, help='Normalize weights')
    parser.add_argument('-p', '--pool_size', type=int, required=True, help='Adaptive Pool Size')
    parser.add_argument('-c', '--out_channel', type=int, required=True, help='Out Channel')
    
    parser.add_argument('-r', '--resume', type=int, required=True, help='Continue training')
    parser.add_argument('-pr', '--pretrained', type=str, help='Load pretrained model')

    args = parser.parse_args()

    # INITIALIZATIONS
    n_epochs = args.epoch
    
    w1_loss = 1
    w2_loss = 1
    w3_loss = 1
    
    if args.loss == 'mse':
        weight_loss = nn.MSELoss()
    elif args.loss == 'mae':
        weight_loss = nn.L1Loss()
    elif args.loss == 'huber':
        weight_loss = nn.SmoothL1Loss()
    
    train = DataLoader(Images(args.dataset, 'TRAINING.csv', args.normalize, False), 
                       batch_size=args.batch_size, num_workers=8, shuffle=True)
    
    valid = DataLoader(Images(args.dataset, 'VAL.csv', args.normalize, False), 
                       batch_size=1, num_workers=8, shuffle=False)
    
    
    print("Training on " + str(len(train)*args.batch_size) + " images.")
    print("Validating on " + str(len(valid)) + " images.")

    net = UNet(args.min_neuron, args.pool_size, args.out_channel)
    
    if args.resume == 1:
        pretrained_model = torch.load(glob(args.pretrained)[0])
        net.load_state_dict(pretrained_model["state_dict"])
        start_epoch = pretrained_model["epoch"]
        v_l = pretrained_model["v_l"]
        v_lw = pretrained_model["v_lw"]
        v_lm = pretrained_model["v_lm"]
        v_lj = pretrained_model["v_lj"]

        t_l = pretrained_model["t_l"]
        t_lw = pretrained_model["t_lw"]
        t_lm = pretrained_model["t_lm"]
        t_lj = pretrained_model["t_lj"]
    else:
        start_epoch = 0
        #pretrained_model = torch.load(glob('models/IMDB_MODEL_07102019_054512/*')[0])
        #state_dict = pretrained_model["state_dict"]

        #own_state = net.state_dict()

        #for name, param in state_dict.items():
        #    if name not in own_state:
        #         continue
        #    if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
        #        param = param.data
        #    own_state[name].copy_(param)
            
        v_l = []
        t_l = []
    
        v_lw = []
        v_lm = []
        v_lj = []

        t_lw = []
        t_lm = []
        t_lj = []
    
    SAVE_DIR = 'IMDB_MODEL_WEIGHT_' + datetime.now().strftime("%d%m%Y_%H%M%S") + '/'
    
    MODEL_SETTINGS = {
        'epoch': n_epochs,
        'learning_rate': args.learning_rate,
        'mask_loss_weight': w1_loss,
        'joint_loss_weight': w2_loss,
        'weight_loss_weight': w3_loss,
        'weight_loss': args.loss,
        'batch_size': args.batch_size,
        'dataset': args.dataset,
        'min_neuron': args.min_neuron
    }
    
    LOG_DIR = 'logs/' + SAVE_DIR
    
    try:
        os.makedirs(LOG_DIR)
        np.save(LOG_DIR + 'model_settings.npy', MODEL_SETTINGS)
    except:
        print("Error ! Model exists.")
    
    # Print Number of Parameters
    n_params = 0

    for param in net.parameters():
        n_params += param.numel()
        
    print('Total params:', n_params)
    print('Trainable params:', sum(p.numel() for p in net.parameters() if p.requires_grad))
    print('Non-trainable params:',n_params-sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    # Use GPU
    cuda = torch.cuda.is_available()
    if cuda:
        net = net.cuda()
        
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate)
    
    # MODEL TRAINING

    best_val = np.inf
    best_ep = -1

    for ep in range(start_epoch, start_epoch+n_epochs):

        with tqdm(total=len(train), dynamic_ncols=True) as progress:
            
            loss_ = 0.
            tm_ = 0.
            tj_ = 0.
            tw_ = 0.
                        
            progress.set_description('Epoch: %s' % str(ep+1))

            for idx, batch_data in enumerate(train):
                X, y_mask, y_joint, y_weight = batch_data['img'].cuda(), batch_data['mask'].cuda(), batch_data['joint'].cuda(), batch_data['weight'].cuda()

                optimizer.zero_grad()
                
                mask_o, joint_o, _, weight_o = net(X)
                
                loss_m = w1_loss * (dice_loss(mask_o, y_mask, 0) + dice_loss(mask_o, y_mask, 1))/2
                loss_j = w2_loss * nn.CrossEntropyLoss()(joint_o, y_joint)  
                loss_w = w3_loss * weight_loss(weight_o, y_weight)
                
                loss = loss_m + loss_j + loss_w

                loss.backward()
                optimizer.step()
                
                progress.update(1)
                
                loss_ += loss.item()
                tm_ += loss_m.item()
                tj_ += loss_j.item()
                tw_ += loss_w.item()

                progress.set_postfix(loss=loss_/(idx+1), mask=tm_/(idx+1), joint=tj_/(idx+1), weight=tw_/(idx+1))

            loss_ /= len(train)
            tm_ /= len(train)
            tj_ /= len(train)
            tw_ /= len(train)
                        
        progress.write('Validating ...')
        
        net.eval()
        
        with torch.no_grad():
            
            vl_ = 0.
            vm_ = 0.
            vj_ = 0.
            vw_ = 0.

            for idx, batch_data in enumerate(valid):
                X, y_mask, y_joint, y_weight = batch_data['img'].cuda(), batch_data['mask'].cuda(), batch_data['joint'].cuda(), batch_data['weight'].cuda()

                mask_o, joint_o, _, weight_o = net(X)
                
                val_loss_m = w1_loss * (dice_loss(mask_o, y_mask, 0) + dice_loss(mask_o, y_mask, 1))/2
                val_loss_j = w2_loss * nn.CrossEntropyLoss()(joint_o, y_joint)  
                val_loss_w = w3_loss * weight_loss(weight_o, y_weight)
                
                val_loss = val_loss_m + val_loss_j + val_loss_w

                vl_ += val_loss.item()
                vm_ += val_loss_m.item()
                vj_ += val_loss_j.item()
                vw_ += val_loss_w.item()

            vl_ /= len(valid)
            vm_ /= len(valid)
            vj_ /= len(valid)
            vw_ /= len(valid)
            
        t_l.append(loss_)
        v_l.append(vl_)
        
        v_lw.append(vw_)
        v_lm.append(vm_)
        v_lj.append(vj_)

        t_lw.append(tw_)
        t_lm.append(tm_)
        t_lj.append(tj_)

        if vl_ <= best_val:

            best_val = vl_

            state = {'epoch': ep + 1, 
                     'state_dict': net.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     't_l': t_l,
                     'v_l': v_l,
                     't_lw': t_lw,
                     't_lm': t_lm,
                     't_lj': t_lj,
                     'v_lw': v_lw,
                     'v_lm': v_lm,
                     'v_lj': v_lj
                    }

            if os.path.exists('models/' + SAVE_DIR):
                os.remove('models/' + SAVE_DIR + 'model_ep_{}.pth.tar'.format(best_ep))
            else:
                os.makedirs('models/' + SAVE_DIR)

            torch.save(state, 'models/' + SAVE_DIR + 'model_ep_{}.pth.tar'.format(ep+1))
            best_ep = ep+1
                

        progress.write('T Loss: {:.3f}\nV Loss: {:.3f}\nV Weight: {:.3f}'.format(loss_, vl_, vw_))
            
        net.train()
   
    np.save(LOG_DIR + 't_loss', np.array(t_l))
    np.save(LOG_DIR + 'v_loss', np.array(v_l))
    
    np.save(LOG_DIR + 't_lw', np.array(t_lw))
    np.save(LOG_DIR + 'v_lw', np.array(v_lw))
    
    np.save(LOG_DIR + 't_lm', np.array(t_lm))
    np.save(LOG_DIR + 'v_lm', np.array(v_lm))
    
    np.save(LOG_DIR + 't_lj', np.array(t_lj))
    np.save(LOG_DIR + 'v_lj', np.array(v_lj))