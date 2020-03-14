import NET.MTCNNNet as NET
from train import train
import torch
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Rnet = NET.RNet().to(device)
    train(Rnet,24,device,"Rnet")