import numpy as np
import matplotlib.pyplot as plt
import os

import utils
from models.semi_with_weights import Semi_siamese   ###semi-siamese with weight
#from models.semi_no_weights import Semi_siamese_    ###semi-siamese no weight
#from models.siamese_with_weights import Siamese      ###siamese with weight

import torch
import torch.optim as optim

from misc.metric_tool import ConfuseMatrixMeter
from models.nets import *
from models.losses import get_alpha, softmax_helper, FocalLoss, mIoULoss, mmIoULoss

from misc.logger_tool import Logger, Timer
from utils import de_norm

def CDTrainer():

    #net_G = Semi_siamese_(in_channels=3, out_channels=3, init_features=32)   ###semi-siamese no weight
    #net_G = Siamese(in_channels=3, out_channels=3, init_features=32)   ###siamese with weight
    net_G = Semi_siamese(in_channels=3, out_channels=3, init_features=32)    ###semi-siamese with weight

    optimizer_G = optim.AdamW(net_G.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
    exp_lr_scheduler_G = get_scheduler(optimizer_G,200) ###### lr_decay_iters

    running_metric = ConfuseMatrixMeter(n_class=2)     ###### 2 classes

    trainloader = utils.get_loader(data_name='WHU',img_size=256,batch_size=16,is_train=True,split='train',dataset='CDDataset_WHU')
    valloader = utils.get_loader(data_name='WHU',img_size=256,batch_size=16,is_train=False,split='val',dataset='CDDataset_WHU')

    dataloaders = {'train':trainloader, 'val':valloader}
    print(len(dataloaders))

    #  training log
    epoch_acc = 0
    epoch_loss = 0     #### loss

    best_val_loss = 0  ####
    best_val_acc = 0.0

    best_epoch_id = 0
    epoch_to_start = 0
    max_num_epochs = 200

    steps_per_epoch = len(dataloaders['train'])
    total_steps = (max_num_epochs - epoch_to_start)*steps_per_epoch

    G_pred = None
    pred_vis = None
    batch = None
    G_loss = None

    is_training = False
    batch_id = 0
    epoch_id = 0

    alpha           = get_alpha(dataloaders['train']) # calculare class occurences
    print(f"alpha-0 (no-change)={alpha[0]}, alpha-1 (change)={alpha[1]}")
    _pxl_loss  = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha, gamma = 2, smooth = 1e-5)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        #device = "cuda"
        if torch.cuda.device_count() > 1:
            net_G = nn.DataParallel(net_G)
    net_G.to(device)

    print('training from scratch...')

    tri_loss=[]
    tri_acc=[]

    vali_loss=[]
    vali_acc=[]

    for epoch_id in range(epoch_to_start, max_num_epochs):

        ################## train #################
        ##########################################
        train_loss=0.0
        train_steps=0

        running_metric.clear()

        is_training = True
        net_G.train()  # Set model to training mode

        for batch_id, batch in enumerate(dataloaders['train'], 0):
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred = net_G(img_in1, img_in2)            

            #####Optimize
            optimizer_G.zero_grad()

            #####Backward
            gt = batch['L'].to(device).long()

            G_loss = _pxl_loss(G_pred, gt)

            G_loss.backward()
            optimizer_G.step()            

            target = batch['L'].to(device).detach()
            G_pred = G_pred.detach()
            G_pred = torch.argmax(G_pred, dim=1)
            current_score = running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())            

            train_loss += G_loss.item()
            train_steps += 1       

        scores = running_metric.get_scores()
        train_acc = scores['mf1']

        exp_lr_scheduler_G.step()

        print('train:')
        print('epoch',epoch_id)
        print(train_loss / train_steps)
        print(train_acc)

        tri_loss.append(train_loss / train_steps)
        tri_acc.append(train_acc)        


        ################## Eval ##################
        ##########################################
        running_metric.clear()
        is_training = False
        net_G.eval()

        val_loss=0
        val_steps=0        


        # Iterate over data.
        for batch_id, batch in enumerate(dataloaders['val'], 0):
            with torch.no_grad():
                #####Forward
                img_in1 = batch['A'].to(device)
                img_in2 = batch['B'].to(device)
                G_pred= net_G(img_in1, img_in2)

                #####Backward
                gt = batch['L'].to(device).long()

                G_loss = _pxl_loss(G_pred, gt)
                ####

                #####Print statistics
                val_loss += G_loss.cpu().numpy()

                val_steps += 1

            target = batch['L'].to(device).detach()
            G_pred = G_pred.detach()
            G_pred = torch.argmax(G_pred, dim=1)
            current_score = running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())

        scores = running_metric.get_scores()
        val_acc = scores['mf1']
            
        print('validation:')
        print('epoch',epoch_id)
        print(val_loss / val_steps)
        print(val_acc)
        print('\n')
       
        checkpoint_dir = '/home/yun13001/code/change_detection/whu_dataset/Sia_train/checkpoints/'
        name='cv_test/' 
         
        if not os.path.exists(checkpoint_dir+name):
            os.makedirs(checkpoint_dir+name)

        path = 'model.pt'

        #### Save the last epoch model
        torch.save({'epoch':epoch_id,
                    'model_G_state_dict': net_G.state_dict(),
                    'optimizer_G_state_dict':optimizer_G.state_dict(),
                    'exp_lr_scheduler_G_state_dict': exp_lr_scheduler_G.state_dict(),
                    'val_acc': val_acc,
                    },checkpoint_dir+name+path)

        #### save best model 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch_id = epoch_id

            torch.save({'epoch':best_epoch_id,
                    'model_G_state_dict': net_G.state_dict(),
                    'optimizer_G_state_dict':optimizer_G.state_dict(),
                    'exp_lr_scheduler_G_state_dict': exp_lr_scheduler_G.state_dict(),
                    'val_acc': val_acc,
                    },checkpoint_dir+name+'best_model.pt')       


        #### Save all the training and validation results
        np.save(checkpoint_dir+name+'train_loss.npy', tri_loss)
        np.save(checkpoint_dir+name+'train_acc.npy', tri_acc)

        np.save(checkpoint_dir+name+'val_loss.npy', vali_loss) 
        np.save(checkpoint_dir+name+'val_acc.npy', vali_acc)        

