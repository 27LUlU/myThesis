import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from collections import Counter
import random
import csv

random.seed(0)

directory = '/results'

# Get a list of file paths for all CSV files in the directory
csv_files = glob.glob(os.path.join(f'{directory}/processed_data/', "*.csv"))

# Extract the name of the CSV file
csv_name = []
for file_path in csv_files:
    csv_name.append(os.path.splitext(os.path.basename(file_path))[0])

count_csv = Counter([item.split('_')[0] for item in csv_name])

# select individuals that have equal and more than 2 segments
selected_names = []
for name, count in count_csv.items():
    if count >= 2:
        selected_names.append(name)

selected_data = []
for name in selected_names:
    for item in csv_name:
        if item.split('_')[0] == name:
            selected_data.append(item)
        else:
            pass

# randomly select 2 segments
grouped_data = {}
for item in selected_data:
    prefix = item.split('_')[0]
    if prefix not in grouped_data:
        grouped_data[prefix] = []
    grouped_data[prefix].append(item)

selected_samples = []
for prefix, items in grouped_data.items():
    selected_sample = random.sample(items, 2)
    selected_samples.append(selected_sample)
flattened_list = [element for sublist in selected_samples for element in sublist]

# save to file
csv_file = f'{directory}/data_selection/final_csv_names.csv'

with open(csv_file, 'w', newline='') as file:

    writer = csv.writer(file)
    writer.writerow(flattened_list)

print("CSV saves successfully:", csv_file)

# plot the number segments of each individuals
count_csv = sorted(count_csv.items(), key=lambda x: x[1], reverse=True)

names = [item[0] for item in count_csv]
counts = [item[1] for item in count_csv]

plt.figure(figsize=(25, 8))
plt.bar(names, counts, color="#4393C3")
plt.xlabel('Patients', fontsize=15)
plt.ylabel('Day counts', fontsize=15)
plt.xticks(ticks=np.arange(len(names)), labels=np.arange(len(names)))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlim(-1, len(names) - 0.5)
plt.xticks(rotation=45, fontsize=13)
plt.savefig(f'{directory}imgs/daycounts.png')
plt.show()
