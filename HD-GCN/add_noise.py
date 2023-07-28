import os
import csv
import shutil
import numpy as np
from random import choice

csv_path = '/home/prgc/ntu60_noise/val.csv'
out_path = '/home/prgc/ntu60_noise/noisy_val.csv'
replace_rate = 0.2

with open(csv_path, 'r') as csv_file:
    noisy_csv = list(csv.reader(csv_file, delimiter=' '))

mask = np.random.choice([0, 1], size=len(noisy_csv), p=((1 - replace_rate), replace_rate)).astype(np.bool)

for i in range(len(noisy_csv)):
    if mask[i]:
        noisy_csv[i][1] = str(choice([i for i in range(1,61) if i not in [int(noisy_csv[i][1])]])).zfill(2)

with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(noisy_csv)
