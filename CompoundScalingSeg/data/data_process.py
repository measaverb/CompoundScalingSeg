import cv2
import numpy
import csv
import os


mask_root = './mask/'
file_list = os.listdir(mask_root)

with open('train_label.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image', 'label'])

for name in file_list:
    img_arr = cv2.imread(os.path.join(mask_root, name))
    print(name)
    base_name = str(name.split('.')[0])
    base_name = base_name[:-5]

    with open('train_label.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if img_arr.sum()== 0:
            writer.writerow([base_name, '0'])
        else:
            writer.writerow([base_name, '1'])


        
        
