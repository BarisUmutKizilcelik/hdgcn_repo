import os

file1 = '/cvhci/temp/prgc/hdgcn_filtered/statistics/skes_available_name.txt'
file2 = '/cvhci/temp/prgc/hdgcn_filtered/statistics/filtered_skes_avaliable_name.txt'
files_root = '/cvhci/temp/prgc/hdgcn_filtered/statistics'
files = ['camera.txt','label.txt','performer.txt','replication.txt', 'setup.txt']
files_try = ['camera.txt']

def create_acc(acc_file):
    in_path = os.path.join(files_root, acc_file)
    out_path = os.path.join(files_root, 'acc' + acc_file)
    with open(file1, 'r') as f1, open(in_path, 'r') as l1:
        f1_lines = f1.readlines()
        l1_lines = l1.readlines()

    mapping = {}
    for line, label in zip(f1_lines, l1_lines):
        mapping[line.strip()] = label.strip()
    print(mapping['S005C001P021R001A016'])

    with open(file2, 'r') as f2:
        f2_lines = f2.readlines()

    l2_lines = [mapping[line.strip()] for line in f2_lines]

    with open(out_path, 'w') as l2:
        l2.write('\n'.join(l2_lines))

for f in files:
    create_acc(f)