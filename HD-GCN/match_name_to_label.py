import os
import csv
import shutil

csv_root = '/home/prgc/ntu60_noise'
files_root = '/cvhci/temp/prgc/hdgcn_filtered/statistics'
file1 = '/cvhci/temp/prgc/hdgcn_filtered/statistics/filtered_skes_avaliable_name.txt'
file2 = '/cvhci/temp/prgc/hdgcn_filtered/statistics/acclabel.txt'
file3 = '/cvhci/temp/prgc/hdgcn_filtered/statistics/noisy_acclabel_20.txt'
names = ['noisy_train_20.csv']

dict = {}
dict_m = {}
for name in names:
    with open(os.path.join(csv_root, name), 'r') as csv_file:
        for line in csv_file:
            line = line.strip().split(' ')
            key = line[0].split('_')[0]
            value = line[1]
            dict[key] = value

with open(file1, 'r') as f1:
        f1_lines = f1.readlines()

with open(file2, 'r') as f2:
    f2_lines = f2.readlines()
    
    for i, line in enumerate(f1_lines):
        if line.strip() in dict.keys():
            dict_m[i] = dict[line.strip()]   


    for i in dict_m:
        f2_lines[i] = dict_m[i] + '\n'
    

with open(file3, 'w') as l1:
    l1.writelines(f2_lines)