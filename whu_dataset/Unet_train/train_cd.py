import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

import os
import copy
import time

def checkpoint(model, optimizer, filename):
    torch.save({
    'optimizer': optimizer.state_dict(),
    'model': model.state_dict(),
}, filename)

def resume(model, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def train_model(model, dataloaders, criterion, optimizer, scheduler, patience, device, num_epochs=25):

    tic = time.perf_counter()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    epoch_list=[]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 1000.0
    best_epoch =0
    iterations = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_pixels = 0

            list_dice_val = []


            # Iterate over data.
            for sample in dataloaders[phase]:
                test_img = sample['A'].to(device)
                labels = (sample['B']).to(device)


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(test_img)                    

                    # Calculate Loss                    
                    loss = criterion(outputs, labels)

                    preds = outputs 

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                running_pixels += torch.sum(labels.data == labels.data)


            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / running_pixels

            print('{} Loss: {:.6f}'.format(phase, epoch_loss))

            # Update Scheduler if training loss doesn't change for patience(2) epochs
            if phase == 'train':
                #sc_plt.step(epoch_loss)
                scheduler.step()


            # deep copy the model and save if accuracy is better
            #if phase == 'val' and epoch_acc > best_acc:
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                print('save at epoch {} with {:f}'.format(best_epoch,epoch_loss))

            # save model 
                checkpoint(model, optimizer, "./save_model/best_model_whu.pth")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

            if phase == 'val':
                val_acc_history.append(epoch_acc.to('cpu'))
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc.to('cpu'))
                train_loss_history.append(epoch_loss)

        ### early stop
        #if epoch - best_epoch > patience:
        #    print("Early stopped training at epoch %d" % epoch)
        #    break  # terminate the training loop


    #print('Best val Acc at {:f} epoch: {:4f}'.format(best_epoch,best_acc))
    print('Best val loss at epoch {}: {:4f}'.format(best_epoch,best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    toc = time.perf_counter()
    print(f"training time {epoch} epochs in {toc - tic:0.4f} seconds")

    return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history, epoch_list



