
# coding: utf-8

# # Use CASIA-SURF training dataset and our private dataset for training

# In[1]:


from pathlib import Path  #从pathlib中导入Path
import os
data_dir = os.getcwd() + '/our_filelist'
txt_dir=[i for i in list(Path(data_dir).glob("**/2*.txt")) ]# 

str1 = '/home/zp/disk1T/CASIASURF/data'
str2 = os.getcwd()
str3 = '/home/zp/disk1T/TSNet-LW/data'

for i in range(len(txt_dir)):
    s = str(txt_dir[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
    s2 = s.replace("'",'').replace('our_filelist','')
    fp = open(s2,'w')
    with open(s,'r') as f:
        lines = f.read().splitlines()
    for i in lines:
        i = i.replace(str1,str2)
        i = i.replace(str3,str2)
        fp.write( i + '\n')
    fp.close()


# # Use CASIA-SURF Val data for val

# In[2]:


import fileinput
rgb = open('./rgb_val.txt','a')
depth = open('./depth_val.txt','a')
ir = open('./ir_val.txt','a')
label = open('./label_val.txt','a')
pwd = os.getcwd() +'/'# the val data path 
for line in fileinput.input("val_label.txt"):
    list = line.split(' ')
    rgb.write(pwd +list[0]+'\n')
    depth.write(pwd +list[1]+'\n')
    ir.write(pwd +list[2]+'\n')
    label.write(list[3])
rgb.close()
depth.close()
ir.close()
label.close()

