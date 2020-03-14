import PIL.Image as pimg
import os
import numpy as np
# from MTCNN.tools import iou
import traceback

save_path = r"F:\数据集\data1"
img_path_dir = r"F:\数据集\img_celeba"
label_path = r"DATA\my_list_bbox_celeba.txt"
def iou(a,b):
    
    score =  (a[2]-a[0])*(a[3]-a[1])
    scores =(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
        
    xx1 = np.maximum(a[0],b[:,0])
    yy1 = np.maximum(a[1],b[:,1])
    xx2 = np.minimum(a[2],b[:,2])
    yy2 = np.minimum(a[3],b[:,3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    scores2 = w*h

    
    return scores2/(score+scores-scores2)

for feature_size in [48]:
    positive_sample_path = os.path.join(save_path,"{}".format(feature_size),"positive")            # 图片存储路径
    if not os.path.exists(positive_sample_path):                                                   # 确保图片路径存在
        os.makedirs(positive_sample_path)

    positive_label_name = os.path.join(save_path,"{}".format(feature_size),"positive_label.txt")   # 图片标签路径

    # 样本数量、样本数量开关
    num_flag = True
    flag = False
    positive_num = 0

    # 文件操作try一下、防止程序崩溃
    try:
        positive_file = open(positive_label_name,"w")     # 以写入的方式打开文件
        while num_flag: # 样本数量开关
            for i, messes in enumerate(open(label_path).readlines()):
                if i < 2:
                    continue
                try:
                    mess = messes.strip().split()
                    img_path = os.path.join(img_path_dir, mess[0])
                    img = pimg.open(img_path)
                    img_w, img_h = img.size
                    x1 = float(mess[1])
                    y1 = float(mess[2])
                    w = float(mess[3])
                    h = float(mess[4])
                    x2, y2 = x1 + w, y1 + h
                    if x1 <= 0 or y1 <= 0 or x2 > img_w or y2 > img_h:
                        continue
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    box = [[x1, y1, x2, y2]]

                    for j in range(10):
                        add_x = np.random.randint(-int(w * 0.2), int(w * 0.2) + 2)
                        add_y = np.random.randint(-int(h * 0.2), int(w * 0.2) + 2)

                        new_side = np.random.randint(int(min(w, h) * 0.6), int(max(w, h) * 1.5) + 2)

                        center_x_ = center_x + add_x
                        center_y_ = center_y + add_y
                        x1_ = center_x_ - new_side / 2
                        y1_ = center_y_ - new_side / 2
                        x2_ = x1_ + new_side
                        y2_ = y1_ + new_side

                        if x1_ <= 0 or y1_ <= 0 or x2_ > img_w or y2_ > img_h or min(img_w,img_h)<40:
                            continue
                        offset_x1 = (x1 - x1_) / new_side        # 实际框和建议框的相对位置关系：建议框的信息已知
                        offset_y1 = (y1 - y1_) / new_side
                        offset_x2 = (x2 - x2_) / new_side
                        offset_y2 = (y2 - y2_) / new_side

                        new_box = [x1_, y1_, x2_, y2_]
                        img_crop = img.crop(new_box)
                        img_resize = img_crop.resize((feature_size, feature_size))

                        iou_ = iou(np.array(new_box), np.array(box))
                        if iou_ > 0.7:
                            positive_file.write(
                                "positive/{}.jpg {} {} {} {} {}\n ".format(positive_num, 1, offset_x1, offset_y1,offset_x2,offset_y2))
                            positive_file.flush()
                            img_resize.save(os.path.join(positive_sample_path, "{}.jpg".format(positive_num)))
                            positive_num += 1
                            if positive_num == 5000:
                                num_flag = False
                                flag = True
                                break
                    if flag:
                        break
                except Exception as e:
                    traceback.print_exc()  # 如果出现异常，把异常打印出来

    finally:
        positive_file.close()













