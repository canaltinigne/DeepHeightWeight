import numpy as np 
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader

from loss import dice_coef, dice_loss
from dataset import Images
from network import UNet
from datetime import datetime
from glob import glob
from torch.nn.parameter import Parameter

"""
Training Height Estimation Network

Example Run:
CUDA_VISIBLE_DEVICES=1 python train.py -e 50 -l 1e-3 -ls mse -d [DIRECTORY OF TRAINING CSV FILE] -r 0
"""
    
if __name__ == "__main__":
    
    # PARSER SETTINGS
    parser = argparse.ArgumentParser(description='U-Net PyTorch Model for Height and Weight Prediction in IMDB Dataset')

    parser.add_argument('-e', '--epoch', type=int, required=True, help='Number of Epochs')
    parser.add_argument('-l', '--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('-ls', '--loss', type=str, required=True, help='Height loss type')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Dataset directory')

    parser.add_argument('-w1', '--w1_loss', type=float, default=1, help='Loss weight for Height Estimation')
    parser.add_argument('-w2', '--w2_loss', type=float, default=1, help='Loss weight for Cross Entropy')
    parser.add_argument('-w3', '--w3_loss', type=float, default=1, help='Loss weight for Dice Loss')
    
    parser.add_argument('-m', '--min_neuron', type=int, default=128, help='Minimum neuron number for the first layer')
    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('-r', '--resume', type=int, required=True, help='Continue training')
    parser.add_argument('-pr', '--pretrained', type=str, help='Load pretrained model')

    args = parser.parse_args()

    # INITIALIZATIONS
    n_epochs = args.epoch
    
    w1_loss = args.w1_loss
    w2_loss = args.w2_loss
    w3_loss = args.w3_loss
        
    
    if args.loss == 'mse':
        height_loss = nn.MSELoss()
    elif args.loss == 'mae':
        height_loss = nn.L1Loss()
    elif args.loss == 'huber':
        height_loss = nn.SmoothL1Loss()
    
    train = DataLoader(Images(args.dataset, 'TRAINING.csv', True), 
                       batch_size=args.batch_size, num_workers=8, shuffle=True)
    
    valid = DataLoader(Images(args.dataset, 'VAL.csv', True), 
                       batch_size=1, num_workers=8, shuffle=False)
    
    
    print("Training on " + str(len(train)*args.batch_size) + " images.")
    print("Validating on " + str(len(valid)) + " images.")

    net = UNet(args.min_neuron)
    start_epoch = 0
    
    #pretrained_model = torch.load(glob('models/IMDB_MODEL_06102019_121502/*')[0])
    #state_dict = pretrained_model["state_dict"]
    
    #own_state = net.state_dict()
    
    #for name, param in state_dict.items():
    #    if name not in own_state:
    #         continue
    #    if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
    #        param = param.data
            
    #    if not (("height_1" in name) or ("height_2" in name)):
    #        own_state[name].copy_(param)
        
    #for param in net.parameters():
    #    param.requires_grad = False
       
    #net.conv_out.height_1[0].weight.requires_grad = True
    #net.conv_out.height_1[0].bias.requires_grad = True
    #net.conv_out.height_2[0].weight.requires_grad = True
    #net.conv_out.height_2[0].bias.requires_grad = True
    #net.conv_out.height_2[3].weight.requires_grad = True
    #net.conv_out.height_2[3].bias.requires_grad = True
    
    SAVE_DIR = 'IMDB_MODEL_' + datetime.now().strftime("%d%m%Y_%H%M%S") + '/'
    
    MODEL_SETTINGS = {
        'epoch': n_epochs,
        'learning_rate': args.learning_rate,
        'mask_loss_weight': w1_loss,
        'joint_loss_weight': w2_loss,
        'height_loss_weight': w3_loss,
        'height_loss': args.loss,
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
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
 
    # MODEL TRAINING
    
    if args.resume == 1:
        pretrained_model = torch.load(glob(args.pretrained)[0])
        net.load_state_dict(pretrained_model["state_dict"])
        start_epoch = pretrained_model["epoch"]
        v_l = pretrained_model["v_l"]
        v_lh = pretrained_model["v_h"]
        v_lm = pretrained_model["v_m"]
        v_lj = pretrained_model["v_j"]

        t_l = pretrained_model["t_l"]
        t_lh = pretrained_model["t_h"]
        t_lm = pretrained_model["t_m"]
        t_lj = pretrained_model["t_j"]
    else:
        v_l = []
        v_lh = []
        v_lm = []
        v_lj = []

        t_l = []
        t_lh = []
        t_lm = []
        t_lj = []
        
        t_acc = []
        v_acc = []

    best_val = np.inf
    best_ep = -1
    
    for ep in range(start_epoch, start_epoch+n_epochs):

        with tqdm(total=len(train), dynamic_ncols=True) as progress:
            
            loss_ = 0.
            tm_ = 0.
            tj_ = 0.
            th_ = 0.
            acc_ = 0.
            
            progress.set_description('Epoch: %s' % str(ep+1))

            for idx, batch_data in enumerate(train):
                X, y_mask, y_joint, y_height = batch_data['img'].cuda(), batch_data['mask'].cuda(), batch_data['joint'].cuda(), batch_data['height'].cuda()

                optimizer.zero_grad()
                
                mask_o, joint_o, height_o = net(X)
                
                loss_m = w1_loss * (dice_loss(mask_o, y_mask, 0) + dice_loss(mask_o, y_mask, 1))/2
                loss_j = w2_loss * nn.CrossEntropyLoss()(joint_o, y_joint)  
                loss_h = w3_loss * height_loss(height_o, y_height)

                loss = loss_h + loss_m + loss_j  
                
                pred = torch.argmax(height_o, 1)
                
                loss.backward()
                optimizer.step()
                
                progress.update(1)
                
                loss_ += loss.item()
                tm_ += loss_m.item()
                tj_ += loss_j.item()
                th_ += loss_h.item()
                
                progress.set_postfix(loss=loss_/(idx+1), mask=tm_/(idx+1), joint=tj_/(idx+1), height=th_/(idx+1), acc=acc_/(idx+1))

            loss_ /= len(train)
            tm_ /= len(train)
            tj_ /= len(train)
            th_ /= len(train)
            acc_ /= len(train)
            
        progress.write('Validating ...')
        
        net.eval()
        
        with torch.no_grad():
            
            vl_ = 0.
            vm_ = 0.
            vj_ = 0.
            vh_ = 0.
            val_acc_ = 0.

            for idx, batch_data in enumerate(valid):
                X, y_mask, y_joint, y_height = batch_data['img'].cuda(), batch_data['mask'].cuda(), batch_data['joint'].cuda(), batch_data['height'].cuda()
              
                mask_o, joint_o, height_o = net(X)

                val_loss_m = w1_loss * (dice_loss(mask_o, y_mask, 0) + dice_loss(mask_o, y_mask, 1))/2
                val_loss_j = w2_loss * nn.CrossEntropyLoss()(joint_o, y_joint)
                val_loss_h = w3_loss * height_loss(height_o, y_height)
                
                val_loss = val_loss_h + val_loss_m + val_loss_j + 
                
                pred = torch.argmax(height_o, 1)

                vl_ += val_loss.item()
                vm_ += val_loss_m.item()
                vj_ += val_loss_j.item()
                vh_ += val_loss_h.item()

            vl_ /= len(valid)
            vm_ /= len(valid)
            vj_ /= len(valid)
            vh_ /= len(valid)
            
        t_l.append(loss_)
        t_lm.append(tm_)
        t_lj.append(tj_)
        t_lh.append(th_)

        v_l.append(vl_)
        v_lm.append(vm_)
        v_lj.append(vj_)
        v_lh.append(vh_)

        if vl_ < best_val:

            best_val = vl_

            state = {'epoch': ep + 1, 
                     'state_dict': net.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     't_l': t_l,
                     't_m': t_lm,
                     't_j': t_lj,
                     't_h': t_lh,
                     'v_l': v_l,
                     'v_m': v_lm,
                     'v_j': v_lj,
                     'v_h': v_lh
                    }

            if os.path.exists('models/' + SAVE_DIR):
                os.remove('models/' + SAVE_DIR + 'model_ep_{}.pth.tar'.format(best_ep))
            else:
                os.makedirs('models/' + SAVE_DIR)

            torch.save(state, 'models/' + SAVE_DIR + 'model_ep_{}.pth.tar'.format(ep+1))
            best_ep = ep+1
                

        progress.write('T Loss: {:.3f} - T Mask: {:.3f} - T Joint: {:.3f} - T Height: {:.3f}\nV Loss: {:.3f} - V Mask: {:.3f} - V Joint: {:.3f} - V Height: {:.3f}'.format(loss_, tm_, tj_, th_, vl_, vm_, vj_, vh_))
            
        net.train()
   
    np.save(LOG_DIR + 't_loss', np.array(t_l))
    np.save(LOG_DIR + 't_mask', np.array(t_lm))
    np.save(LOG_DIR + 't_joint', np.array(t_lj))
    np.save(LOG_DIR + 't_height', np.array(t_lh))
    
    np.save(LOG_DIR + 'v_loss', np.array(v_l))
    np.save(LOG_DIR + 'v_mask', np.array(v_lm))
    np.save(LOG_DIR + 'v_joint', np.array(v_lj))
    np.save(LOG_DIR + 'v_height', np.array(v_lh))
    
    state = {'epoch': start_epoch+n_epochs, 
             'state_dict': net.state_dict(),
             'optimizer': optimizer.state_dict(),
             't_l': t_l,
             't_m': t_lm,
             't_j': t_lj,
             't_h': t_lh,
             'v_l': v_l,
             'v_m': v_lm,
             'v_j': v_lj,
             'v_h': v_lh
    }

    torch.save(state, 'models/' + SAVE_DIR + 'last_model_ep_{}.pth.tar'.format(start_epoch+n_epochs))