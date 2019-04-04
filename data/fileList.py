
# coding: utf-8

# # Use CASIA-SURF training dataset and our private dataset for training

# In[1]:


from pathlib import Path  #从pathlib中导入Path
import os
# data_dir = os.getcwd() + '/our_filelist'
# txt_dir=[i for i in list(Path(data_dir).glob("**/2*.txt")) ]# 

# Use CASIA-SURF traing data and our private data
# str1 = '/home/zp/disk1T/CASIASURF/data'
# str2 = os.getcwd()
# str3 = '/home/zp/disk1T/TSNet-LW/data'

# for i in range(len(txt_dir)):
#     s = str(txt_dir[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
#     s2 = s.replace("'",'').replace('our_filelist','')
#     fp = open(s2,'w')
#     with open(s,'r') as f:
#         lines = f.read().splitlines()
#     for i in lines:
#         i = i.replace(str1,str2)
#         i = i.replace(str3,str2)
#         fp.write( i + '\n')
#     fp.close()


# # Use CASIA-SURF Val data for val

# Use CASIA-SURF training data for training

import fileinput
rgb = open('./rgb_train.txt','a')
depth = open('./depth_train.txt','a')
ir = open('./ir_train.txt','a')
label = open('./label_train.txt','a')
pwd = os.getcwd() +'/'# the val data path 
for line in fileinput.input("train_list.txt"):
    list = line.split(' ')
    rgb.write(pwd +list[0]+'\n')
    depth.write(pwd +list[1]+'\n')
    ir.write(pwd +list[2]+'\n')
    label.write(list[3])
rgb.close()
depth.close()
ir.close()
label.close()

import fileinput
rgb = open('./rgb_val.txt','a')
depth = open('./depth_val.txt','a')
ir = open('./ir_val.txt','a')
label = open('./label_val.txt','a')
pwd = os.getcwd() +'/'# the val data path 
for line in fileinput.input("val_private_list.txt"):
    list = line.split(' ')
    rgb.write(pwd +list[0]+'\n')
    depth.write(pwd +list[1]+'\n')
    ir.write(pwd +list[2]+'\n')
    label.write(list[3])
rgb.close()
depth.close()
ir.close()
label.close()

# Use CASIA-SURF Test data for test
# To make it easier for you to test, prepare the label for the test set.
import fileinput
rgb = open('./rgb_test.txt','a')
depth = open('./depth_test.txt','a')
ir = open('./ir_test.txt','a')
label = open('./label_test.txt','a')
pwd = os.getcwd() +'/'# the val data path 
for line in fileinput.input("test_private_list.txt"):
    list = line.split(' ')
    rgb.write(pwd +list[0]+'\n')
    depth.write(pwd +list[1]+'\n')
    ir.write(pwd +list[2]+'\n')
    label.write(list[3])
rgb.close()
depth.close()
ir.close()
label.close()

# In test phase，we use the IR data for training
# replace '/home/zp/disk1T/libxcam-testset/' 
f = open('ir_final_train.txt','w')
ir_file = 'ir_final_train_tmp.txt'
s = '/home/zp/disk1T/libxcam-testset/data'
import os
dir_pwd = os.getcwd() 
with open(ir_file,'r') as fp:
    lines = fp.read().splitlines()
    for line in lines:
        line = line.replace(s,dir_pwd)
        f.write(line + '\n')
f.close()