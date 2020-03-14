import PIL.Image as Image
import NET.MTCNNNet as NET
import torchvision.transforms as T
import PIL.ImageDraw as Draw
import torch
totensor = T.Compose([
        T.Resize((12, 12)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
net = NET.PNet().eval()
net.load_state_dict(torch.load("model/Pnet.ph"))
img = Image.open(r"F:\数据集\img_celeba\000103.jpg")
w, h =48,48#img.size
f = T.Resize((48, 48))
img = f(img)
draw = Draw.ImageDraw(img)
b,a =net(totensor(img).reshape(1,3,12,12))
a = a.reshape(-1,4)[0]
print(b)
draw.rectangle((a[0]*w,a[1]*h,a[2]*w,a[3]*h),fill=None,outline="blue",width=1)
 
img.show()