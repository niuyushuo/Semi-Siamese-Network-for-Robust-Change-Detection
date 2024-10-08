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

def CDEva():

    #net_G = Semi_siamese_(in_channels=3, out_channels=3, init_features=32)   ###semi-siamese no weight
    #net_G = Siamese(in_channels=3, out_channels=3, init_features=32)   ###siamese with weight
    net_G = Semi_siamese(in_channels=3, out_channels=3, init_features=32)    ###semi-siamese with weight    

    running_metric = ConfuseMatrixMeter(n_class=2)     ###### 2 classes

    trainloader = utils.get_loader(data_name='WHU',img_size=256,batch_size=1,is_train=True,split='train',dataset='CDDataset_WHU')
    valloader = utils.get_loader(data_name='WHU',img_size=256,batch_size=1,is_train=False,split='val',dataset='CDDataset_WHU')  
    testloader = utils.get_loader(data_name='WHU',img_size=256,batch_size=1,is_train=False,split='test',dataset='CDDataset_WHU')

    dataloaders = {'train':trainloader, 'val':valloader, 'test':testloader}    

    #PATH = '/home/yun13001/code/change_detection/whu_dataset/Sia_train/checkpoints/cv_test/best_model.pt'
    PATH = '/home/yun13001/code/change_detection/whu_dataset/save_model/Semi_weights/WHU_test_bce_semi/best_ckpt.pt'

    #### single gpu load model
    checkpoint = torch.load(PATH,weights_only=False)
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    #epoch_save = checkpoint['epoch']    

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net_G = torch.nn.DataParallel(net_G)
    net_G.to(device)
    net_G.eval()
    """
    ##### multiple gpus load model
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            net_G = torch.nn.DataParallel(net_G)
    net_G.to(device)
    checkpoint = torch.load(PATH,weights_only=False)
    net_G.load_state_dict(checkpoint['model_G_state_dict'])
    epoch_save = checkpoint['epoch']
    net_G.eval()
    """
    ground_truth=[]    
    prediction=[]

    # Iterate over data.
    #for batch_id, batch in enumerate(dataloaders['val'], 0):
    for batch_id, batch in enumerate(dataloaders['test'], 0):    
        with torch.no_grad():
            #####Forward
            img_in1 = batch['A'].to(device)
            img_in2 = batch['B'].to(device)
            G_pred = net_G(img_in1, img_in2)

            #####Backward
            gt = batch['L'].to(device).long()    

            pred = torch.argmax(G_pred, dim=1, keepdim=True)
            pred = utils.make_numpy_grid(pred) 

            vis_input = utils.make_numpy_grid(de_norm(batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(batch['B']))

            gt = utils.make_numpy_grid(gt)            
            
            #print('shape test:')
            #print(vis_input.shape)
            #print(vis_input2.shape)
            #print(pred.shape)
            #print(gt.shape)

            vis = np.concatenate([vis_input, vis_input2, pred, gt], axis=1)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)

            vis_dir = '/home/yun13001/code/change_detection/whu_dataset/Sia_train/vis'
            file_name = os.path.join(vis_dir, 'eval_' + str(batch_id)+'.jpg')
            plt.imsave(file_name, vis)   

        target = batch['L'].to(device).detach()
        G_pred = G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        current_score = running_metric.undate_score(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())   ### all metrics 

        message = '%s: ,' % (batch_id)
        for k, v in current_score.items():
            message += '%s: %.5f ' % (k, v)
        print(message)            

    scores = running_metric.get_scores()
    val_acc = scores['mf1']

    message = 'Total:'
    for k, v in scores.items():
        message += '%s: %.5f ' % (k, v)
    print(message)    
