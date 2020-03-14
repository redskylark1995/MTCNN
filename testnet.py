import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import DATA.celebadata as data
import NET.MTCNNNet as NET
import torchvision.transforms as T
import time
import PIL.Image as pimg
import PIL.ImageDraw as Draw
from TOOL.nms import nms
import numpy as np

net  = NET.ONet().eval()
# if os.path.exists(r"model2/"+name+".ph"):
net.load_state_dict(torch.load(r"model2\Onet.pth"))
transform  = T.Compose([
        T.Resize((48, 48)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
img = pimg.open(r"test\000822.jpg.1.0.jpg")#C:\Users\Administrator\Downloads\psb.jpg
outcond, outputs = net(transform(img).reshape(1,3,48,48))
x1 = outputs[0][0]*48
x2 = outputs[0][1]*48
x3 = outputs[0][2]*48
x4 = outputs[0][3]*48
dr = Draw.ImageDraw(img)
print(x1,x2,x3,x4)
dr.rectangle((x1,x2,x3,x4),fill=None,outline="blue",width=1)
img.show()