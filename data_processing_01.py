import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import re
from hees_2013 import hees_2013_calculate_non_wear_time

def missing_process(selected_df):
    # calculate the difference between current second and previous second
    ts = [ts_elem.to_pydatetime() for ts_elem in selected_df['time']]
    ts = [t.strftime('%Y-%m-%d %H:%M:%S') for t in ts]
    ts = pd.to_datetime(ts).unique()
    ts = pd.DataFrame(ts, columns=['time'])
    ts['diff'] = ts['time'].diff().dt.total_seconds()

    # if difference is not 1, record the missing second
    missing_sec = []
    for i in range(1, len(ts)):
        if ts['diff'][i] != 1:
            missing_sec.append(pd.date_range(start=ts['time'][i - 1], end=ts['time'][i], freq='S')[1:-1])
    missing_sec = [item for sublist in missing_sec for item in sublist]

    missing_sec_list = [value for sublist in [[elem] * 30 for elem in missing_sec] for value in sublist]
    timestamp_microseconds = [int(sec.timestamp() * 1e6) for sec in missing_sec_list]

    # interpolate 30 sample datetimes in one second
    initial_value = [3.3333, 3.3334, 3.3333]
    interpolation_values = []
    cumulative_sum = 0
    missing_result = []

    for i in range(0, len(timestamp_microseconds)):
        if i % 30 == 0:
            interpolation_values.append(0)
        else:
            index = (i % 3)
            interpolation_values.append(initial_value[index - 1])

    for i in range(len(timestamp_microseconds)):
        cumulative_sum = interpolation_values[i] + cumulative_sum
        if i % 30 == 0:
            cumulative_sum = 0
        next_microsecond = timestamp_microseconds[i] + cumulative_sum * 10000
        missing_result.append(next_microsecond)

    missing_result = pd.to_datetime(missing_result, unit='us')
    missing_result = pd.DataFrame({'time': missing_result})
    missing_result['X'] = 0
    missing_result['Y'] = 0
    missing_result['Z'] = 0

    complete_sec = pd.concat([selected_df[['time', 'X', 'Y', 'Z']], missing_result], ignore_index=True)
    complete_sec = complete_sec.sort_values(by='time').reset_index(drop=True)

    return complete_sec


# Directory containing the CSV files
directory = '/raw_data'


# Get a list of file paths for all CSV files in the directory
csv_files = glob.glob(os.path.join(f'{directory}', "*.csv"))

# Read metadata
metadata = pd.read_excel('metadata.xlsx')
metadata['tSTART'] = pd.to_timedelta(metadata['tSTART'])
metadata['tSTOP'] = pd.to_timedelta(metadata['tSTOP'])


# Iterate over each CSV file
for file_path in csv_files:
    # Extract the name of the CSV file
    csv_name = os.path.splitext(os.path.basename(file_path))[0]
    num = re.findall(r'\d+', csv_name)
    if len(num) > 1:
        num = num[1]
    else:
        num = num[0]
    csv_name_new = f'ID{num}'

    # select start recording time and stop recording time based on the record in metadata
    if csv_name_new in metadata['ID'].values:
        index = metadata[metadata['ID'] == csv_name_new].index[0]
        start_date = metadata['START'][index]
        start_time = metadata['tSTART'][index]
        stop_date = metadata['STOP'][index]
        stop_time = metadata['tSTOP'][index]

    start_date_time = start_date + start_time
    stop_date_time = stop_date + stop_time


    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    selected_df = df[(df['time'] >= start_date_time) & (df['time'] <= stop_date_time)].reset_index(drop=True)

    # checking missing seconds and filling to continuous time series
    complete_df = missing_process(selected_df)
    complete_np = complete_df[['X', 'Y', 'Z']].values

    # checking wear and non-wear time
    non_wear_detect = hees_2013_calculate_non_wear_time(data=complete_np, hz=30)
    complete_df['wear_or_not'] = non_wear_detect
    start_index = None
    spans = []

    # Loop through the DataFrame to find consecutive spans of 1 in column 'wear_or_not'
    for i in range(len(complete_df)):
        if complete_df.loc[i, 'wear_or_not'] == 1:
            if start_index is None:
                start_index = i
        else:
            if start_index is not None:
                spans.append((start_index, i - 1))
                start_index = None
        print(i)

    # If the last span goes till the end of the DataFrame
    if start_index is not None:
        spans.append((start_index, len(complete_df) - 1))

    # select one day's time equal or larger than 8 hours
    data_time = [sublist[1] - sublist[0] for sublist in spans]
    valid_time = 30 * 60 * 60 * 8  # 8 hours data
    data_time = np.array(data_time)
    day_index = np.where(data_time >= valid_time)[0]

    # save results based on one day
    for i in day_index:
        start = spans[i][0]
        stop = spans[i][1]
        complete_df.loc[start:stop].to_csv(f'/results/processed_data/{csv_name_new}_{i}.csv', index=False)

        plt.plot(complete_df['X'][start:stop])
        plt.savefig(f'/results/imgs/{csv_name_new}_{i}.png')
        plt.close()

