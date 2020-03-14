import torch
import os
import torchvision.transforms as T
from torch.utils.data import DataLoader 
import PIL.Image as pimg
 
class MyDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self,datapath,labepath,transform):
        super(MyDataset, self).__init__()
        self.datapath = datapath
        self.labepath = open(labepath).readlines()
        self.T = transform
        # self.addindex = addindex


    def __getitem__(self, index):
        # index = index + self.addindex
        labe = self.labepath[index].strip().split(" ")
        imgdata = self.T(pimg.open(os.path.join(self.datapath,labe[0])))
        cond = torch.Tensor([int(labe[5])])
        offset = torch.Tensor([int(labe[1]),int(labe[2]),int(labe[3]),int(labe[4])])
        return imgdata,cond,offset

    def __len__(self):
        return len(self.labepath)

# class lhyDataset(torch.utils.data.Dataset):
#     """Some Information about MyDataset"""
#     def __init__(self,datapath,labepath,transform):
#         super(lhyDataset, self).__init__()
#         self.datapath = datapath
#         self.labepath = open(labepath).readlines()
#         self.T = transform


#     def __getitem__(self, index):
#         labe = self.labepath[index].strip().split(" ")
#         # print(labe[0])
#         try:
#             imgdata = self.T(pimg.open(os.path.join(self.datapath,labe[0])))
#         except expression as identifier:
#             print(index)
       
#         cond = torch.Tensor([int(labe[1])])
#         offset = torch.Tensor([float(labe[2]),float(labe[3]),float(labe[4]),float(labe[5])])
#         return imgdata,cond,offset

#     def __len__(self):
#         return len(self.labepath)
if __name__ =="__main__":
    transform = T.Compose([
        T.Resize((48,48)),
        T.ToTensor(),
    # T.Normalize(0.5,0.5)
    ])
    dataset = MyDataset(r"F:\数据集\newimg_celeba1",
    r"D:\VSCodeProjects\人脸追踪\DATA\my_list_bbox_celeba_1.txt",
    transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i, data in enumerate(dataloader, 0):
        print(data[3])
        break

