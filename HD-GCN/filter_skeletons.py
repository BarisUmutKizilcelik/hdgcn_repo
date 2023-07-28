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
                skeleton = skeleton + '.skeleton'
                skeletons.append(skeleton)

for skeleton in skeletons:
    src_file_path = os.path.join(skeletons_path, skeleton)
    dest_file_path = os.path.join(filtered_skeletons_path, skeleton)
    if os.path.isfile(src_file_path):
        shutil.copy2(src_file_path, dest_file_path)
    else:
        print('File not found')

if len(skeletons) == len(os.listdir(filtered_skeletons_path)):
    print('----------------')
    print('All done')