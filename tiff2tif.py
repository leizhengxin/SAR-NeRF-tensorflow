import os 

path = "dataset/angle_1_train"
for i in os.listdir(path):
    if i.endswith(".tiff"):
        os.rename(os.path.join(path,i),os.path.join(path,os.path.splitext(i)[0]+".tif"))