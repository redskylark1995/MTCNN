import NET.MTCNNNet as NET
from train import train
import torch
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Pnet = NET.PNet().to(device)
    train(Pnet,12,device,"Pnet")