import os
import csv
import shutil

csv_path = '/home/prgc/ntu60_rgb'
skeletons_path = '/cvhci/data/activity/NTU_RGBD/zipped/zipped_skeleton_csv/NTU120_skeleton/nturgb+d_skeletons'
filtered_skeletons_path = '/cvhci/temp/prgc/filtered_skeletons'
skeletons = []

for file_name in os.listdir(csv_path):
    if file_name.endswith('.csv'):
        with open(os.path.join(csv_path, file_name), 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for line in csv_reader:
                skeleton = line[0].split('_')[0]
                skeletons.append(skeleton)

with open('/home/prgc/acrionreco-with-noisy-data-topic-c/HD-GCN/data/ntu120/statistics/filtered_skes_avaliable_name.txt', 'w') as txt_file:
    for skeleton in skeletons:
        txt_file.write(skeleton + '\n')