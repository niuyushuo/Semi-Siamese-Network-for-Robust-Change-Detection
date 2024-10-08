from argparse import ArgumentParser
import torch
from models.train_sia import*

#print(torch.cuda.is_available())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CDTrainer()
