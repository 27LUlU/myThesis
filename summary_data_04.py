import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

directory = '/results'

df = pd.read_csv(f'{directory}/feature_extraction/final_feature_table.csv', header=None)
df = df.rename(columns={0: 'ID'})
df['ID_prefix'] = df['ID'].str.split('_').str[0]

# import metadata to get labels
metadata = pd.read_excel('metadata.xlsx')
metadata['label'] = [0 if v == 4 else 1 for v in metadata['ITEM8']]

id_names = np.unique([name.split('_')[0] for name in df['ID']])
id_labels = pd.DataFrame(columns=['ID', 'label'])
for id in id_names:
    id_labels = id_labels.append(metadata[metadata['ID'] == id][['ID', 'label']], ignore_index=True)

# to show the distribution of labels
# count_labels = Counter(id_labels['label'])
# count_labels = sorted(count_labels.items())
# label_values = [t[0] for t in count_labels]
# count_values = [t[1] for t in count_labels]
#
# count_values[3] / sum(count_values)
# plt.bar(label_values, count_values)
# plt.xlabel('labels')
# plt.ylabel('counts')
# plt.title('The distribution of labels')
# plt.show()

# # Iterate over each CSV file to calculate mean and std
df_avg = df.groupby('ID_prefix').mean().reset_index(drop=False)
df_avg_all = pd.merge(df_avg, metadata[['ID', 'label']], left_on='ID_prefix', right_on='ID', how='inner')
df_avg_all = df_avg_all.drop(columns=['ID_prefix']).reset_index(drop=True)
df_avg_all.to_csv(f'{directory}/split_data/df_avg.csv', index=False)
