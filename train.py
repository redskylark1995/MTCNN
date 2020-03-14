import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import DATA.celebadata as data1
import torchvision.transforms as T
from torch.utils.data import DataLoader 
import NET.MTCNNNet as NET
import numpy as np
import os

from multiprocessing import Pool



def train(net,size,device,name):
    print("开始了")
    atransform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = data1.MyDataset(r"C:\新建文件夹\data",r"DATA\my_list_bbox_celeba.txt",transform=atransform)
    # dataset = data1.lhyDataset(r"F:\数据集\data1\48",r"F:\数据集\data1\48\labex.txt",transform=atransform)
    dataloader = DataLoader(dataset, batch_size=1024,shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    criterionBCE = nn.BCELoss()
    criterionMSE = nn.MSELoss()

    if os.path.exists(r"model/"+name+".pth"):
        net.load_state_dict(torch.load(r"model/"+name+".pth"))
    epoch = 0
    # loop over the dataset multiple times
    minloss = 0.2
    while True:
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, cond,labels = data
            condindex =np.where(cond.numpy()<2)[0]
            offsetindex = np.where(cond.numpy()>0)[0]
            inputs, condcuda,labelscuda = inputs.to(device), cond.to(device),(labels/48).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # # forward + backward + optimize
            outcond,outputs = net(inputs)

            outcond = outcond.view(-1, 1)
            outputs = outputs.view(-1,4)
            # print(outputs.shape)
            loss1 = criterionMSE(outputs[offsetindex], labelscuda[offsetindex])

            loss2 = criterionBCE(outcond[condindex],condcuda[condindex])
            loss = loss1+loss2
            
            if loss.item()>100:
                print("损失要飞")
                # log = open("log.log","a")
                

                # break
                print(loss.item())
                # break
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i%10 == 0:
                # print("---------------------------")
                print("\n",name,"epoch",epoch,"i",i,"loss",loss.item())
                print("condloss",loss2.item(),"\ncord",loss1.item())
                print("minloss",minloss)
                print("running_loss/(i+1)",running_loss/(i+1))
                if running_loss/(i+1)<minloss and loss.item()< minloss:
                    
                    minloss = running_loss/(i+1)
                    torch.save(net.state_dict(),r"model/"+name+".pth")
                    print("save net dict")
            # break
            
        epoch +=1
        print('Loss: {}'.format(running_loss/(i+1)))
        
    print('Finished Training')
