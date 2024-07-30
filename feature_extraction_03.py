import numpy as np
import pandas as pd
import os
import glob
import extract_ac
import matplotlib.pyplot as plt
import datetime
import numpy.fft as fft
import csv

directory = '/results'

# Get a list of file paths for all CSV files in the directory
csv_name = pd.read_csv(f'{directory}/data_selection/final_csv_names.csv')
csv_files = glob.glob(os.path.join(f'{directory}/processed_data/', "*.csv"))

# initial time column name
time_column = 'time'

def get_counts_csv(
    raw,
    freq: int,
    epoch: int,
    fast: bool = True,
    verbose: bool = False,
    time_column: str = None,
):

    if time_column is not None:
        ts = pd.to_datetime(raw[time_column])
        time_freq = str(epoch) + "S"
        ts = ts - pd.to_timedelta(epoch / 2, unit='s')
        ts = ts.dt.round(time_freq)
        ts = ts.unique()
        ts = pd.DataFrame(ts, columns=[time_column])

    raw_clean = raw[["X", "Y", "Z"]]
    if verbose:
        print("Converting to array", flush=True)
    raw_clean = np.array(raw_clean)
    if verbose:
        print("Getting Counts", flush=True)
    counts = extract_ac.get_counts(raw_clean, freq=freq, epoch=epoch, fast=fast)
    del raw_clean
    counts = pd.DataFrame(counts, columns=["Axis1", "Axis2", "Axis3"])
    counts["AC"] = (
        counts["Axis1"] ** 2 + counts["Axis2"] ** 2 + counts["Axis3"] ** 2
    ) ** 0.5

    ts = ts[0: counts.shape[0]]
    if time_column is not None:
        counts = pd.concat([ts, counts], axis=1)
    return counts

# feature: %active
def active(counts):
    counts_minute = counts.groupby([counts[time_column].dt.date, counts[time_column].dt.hour, counts[time_column].dt.minute]).sum()

    counts_minute_ts = counts_minute.index.values
    counts_minute_ts = [datetime.datetime.combine(i[0], datetime.time(i[1], i[2])) for i in counts_minute_ts]
    counts_minute_ts = [t.strftime('%Y-%m-%d %H:%M') for t in counts_minute_ts]
    counts_minute[time_column] = pd.to_datetime(counts_minute_ts)
    counts_minute = counts_minute.reset_index(drop=True)

    counts_minute['type'] = [1 if c < 100 else (2 if c < 760 else (3 if c < 2020 else 4)) for c in counts_minute['AC']]
    # counts_minute['type'] = [1 if c < 190 else (2 if c < 800 else (3 if c < 2500 else 4)) for c in counts_minute['AC']]
    counts_minute['active'] = ['inactive' if t == 1 else 'active' for t in counts_minute['type']]

    return counts_minute
def active_stat(count_min):
    active_size = count_min.groupby('active').size()
    active_percent = active_size['active'] / active_size.sum()
    return active_percent

# feature: met
def met(counts):
    # The coefficient of variation of the counts per 10 s
    i = 0
    ac_10_cv = []
    ac_10 = []
    while i < len(counts):
        j = i + 10
        ac_10.append(counts['AC'][i:j].sum())
        ac_10_std = counts['AC'][i:j].std()
        ac_10_mean = counts['AC'][i:j].mean()
        if ac_10_mean == 0:
            ac_10_cv.append(0)
        else:
            ac_10_cv.append((ac_10_std / ac_10_mean) * 100)
        i = j
    ac_10_cv = pd.DataFrame({'ac10': ac_10, 'cv10': ac_10_cv})

    # The met of the counts per 10 s
    met_10 = []
    for c in range(len(ac_10_cv)):
        if ac_10_cv['ac10'][c] <= 8:
            met_10.append(1)
        elif ac_10_cv['ac10'][c] > 8 and ac_10_cv['cv10'][c] <= 10:
            met_10.append(2.294275 * (np.exp(0.00084679 * ac_10_cv['ac10'][c])))
        elif ac_10_cv['ac10'][c] > 8 and ac_10_cv['cv10'][c] > 10:
            ln_ac10 = np.log(ac_10_cv['ac10'][c])
            met_10.append(0.749395 + (0.716431 * ln_ac10) - (0.179874 * (ln_ac10 ** 2)) + (0.033173 * (ln_ac10 ** 3)))

    met_10 = pd.DataFrame(met_10)

    # The met of the counts per day
    met_1d = met_10.mean().values[0]
    return met_1d

# feature: daily_vm
def daily_VM(counts):
    # daily VM
    counts_day_std = counts['AC'].std()
    counts_day_mean = counts['AC'].mean()
    vm = np.log(counts_day_std * counts_day_mean + 1)
    return vm

# check if the all the labels in the dictionary. If not, adding the missed label and set value to 0
def missing_key(dict):
    if len(dict) != 4:
        missing_keys = [key for key in [1, 2, 3, 4] if key not in dict.keys()]
        for i in range(len(missing_keys)):
            dict[missing_keys[i]] = 0
    return dict

# feature:
# The accumulated/max/standard deviation/mean/ratio time of each activity type
# (sedentary, light, moderate-to-vigorous, and vigorous) bouts within one segment.
def activity_bout_time(counts_minute):
    activity_arrays = {}
    # Iterate over the types
    for type_name, type_data in counts_minute.groupby('type'):
        # Sort the type_data by the time_column
        type_data = type_data.sort_values(by=time_column).reset_index(drop=False)

        # Initialize lists to store continuous duration intervals
        bout_intervals = []
        current_interval = [type_data.iloc[0]['index']]

        # Iterate over the rows in the type_data
        for i in range(1, len(type_data)):
            # Check if the current row's time is continuous with the previous row
            if (type_data.iloc[i][time_column] - type_data.iloc[i - 1][time_column]).total_seconds() <= 60:
                current_interval.append(type_data.iloc[i]['index'])
            else:
                bout_intervals.append(current_interval)
                current_interval = [type_data.iloc[i]['index']]

        # Append the last interval
        bout_intervals.append(current_interval)

        # Store the continuous intervals in the activity_arrays dictionary
        activity_arrays[type_name] = bout_intervals

    aggregate_minutes_each_time = {}
    max_minutes_each_time = {}
    mean_minutes_each_time = {}
    ratio_minutes_each_time = {}
    std_minutes_each_time = {}
    for type_name in activity_arrays:
        agg_bout_len = []
        for bout in activity_arrays[type_name]:
            bout_len = len(bout)
            agg_bout_len.append(bout_len)
        aggregate_minutes_each_time[type_name] = np.sum(agg_bout_len)
        max_minutes_each_time[type_name] = np.max(agg_bout_len)
        std_minutes_each_time[type_name] = np.std(agg_bout_len)
        mean_minutes_each_time[type_name] = np.sum(agg_bout_len) / len(agg_bout_len)
        ratio_minutes_each_time[type_name] = np.sum(agg_bout_len) / len(counts_minute)

    aggregate_minutes_each_time = missing_key(aggregate_minutes_each_time)
    max_minutes_each_time = missing_key(max_minutes_each_time)
    std_minutes_each_time = missing_key(std_minutes_each_time)
    mean_minutes_each_time = missing_key(mean_minutes_each_time)
    ratio_minutes_each_time = missing_key(ratio_minutes_each_time)

    return aggregate_minutes_each_time, max_minutes_each_time, std_minutes_each_time, mean_minutes_each_time, ratio_minutes_each_time

# The accumulated/max/standard deviation/mean/ratio count of each activity type
# (sedentary, light, moderate-to-vigorous, and vigorous) bouts within one segment.
def activity_bout_count(counts_minute):
    activity_arrays = {}
    # Iterate over the types
    for type_name, type_data in counts_minute.groupby('type'):
        # Sort the type_data by the time_column
        type_data = type_data.sort_values(by=time_column).reset_index(drop=False)

        # Initialize lists to store continuous duration intervals
        bout_intervals = []
        current_interval = [type_data.iloc[0]['AC']]

        # Check if the current row's time is continuous with the previous row
        # Calculate each bout counts
        for i in range(1, len(type_data)):
            if (type_data.iloc[i][time_column] - type_data.iloc[i - 1][time_column]).total_seconds() <= 60:
                current_interval.append(type_data.iloc[i]['AC'])
            else:
                bout_intervals.append(current_interval)
                current_interval = [type_data.iloc[i]['AC']]

        # Append the last interval
        bout_intervals.append(current_interval)

        # Store the continuous intervals in the activity_arrays dictionary
        activity_arrays[type_name] = bout_intervals

    aggregate_minutes_each_activity = {}
    max_minutes_each_activity = {}
    mean_minutes_each_activity = {}
    ratio_minutes_each_activity = {}
    std_minutes_each_activity = {}
    for type_name in activity_arrays:
        agg_bout_len = []
        mean_bout_len = []
        for bout in activity_arrays[type_name]:
            agg_bout_len.append(np.sum(bout))
            mean_bout_len.append(np.sum(bout)/len(bout))
        aggregate_minutes_each_activity[type_name] = np.sum(agg_bout_len)
        max_minutes_each_activity[type_name] = max(agg_bout_len)
        mean_minutes_each_activity[type_name] = np.mean(mean_bout_len)
        std_minutes_each_activity[type_name] = np.std(mean_bout_len)
        ratio_minutes_each_activity[type_name] = np.sum(agg_bout_len) / counts_minute['AC'].sum()

    aggregate_minutes_each_activity = missing_key(aggregate_minutes_each_activity)
    max_minutes_each_activity = missing_key(max_minutes_each_activity)
    mean_minutes_each_activity = missing_key(mean_minutes_each_activity)
    std_minutes_each_activity = missing_key(std_minutes_each_activity)
    ratio_minutes_each_activity = missing_key(ratio_minutes_each_activity)
    return aggregate_minutes_each_activity, max_minutes_each_activity, mean_minutes_each_activity, std_minutes_each_activity, ratio_minutes_each_activity


# feature:
# mean, standard deviation, coefficient of variation,
# minimum, maximum, 25th, 50th and 75th percentile.
def basic_stats(counts):
    counts_minute_group = counts.groupby([counts[time_column].dt.date, counts[time_column].dt.hour, counts[time_column].dt.minute])

    mean_counts_minutes = counts_minute_group.mean()
    std_counts_minutes = counts_minute_group.std()
    cv_counts_minutes = std_counts_minutes / mean_counts_minutes
    max_counts_minutes = counts_minute_group.max()
    min_counts_minutes = counts_minute_group.min()
    p25_counts_minutes = counts_minute_group.quantile(0.25)
    p50_counts_minutes = counts_minute_group.quantile(0.50)
    p75_counts_minutes = counts_minute_group.quantile(0.75)
    p90_counts_minutes = counts_minute_group.quantile(0.90)
    p95_counts_minutes = counts_minute_group.quantile(0.95)


    # Resample data to 5-minute intervals and calculate sum of acceleration values
    agg_counts_minutes = counts_minute_group.sum()
    count = 0
    agg_counts_5minutes = []
    while (count < len(agg_counts_minutes)):
        agg_counts_5minutes.append(agg_counts_minutes['AC'][count:count+5].max())
        count = count + 5

    max_counts_day_5min = np.mean(agg_counts_5minutes)

    mean_counts_day = mean_counts_minutes['AC'].mean()
    std_counts_day = std_counts_minutes['AC'].mean()
    cv_counts_day = cv_counts_minutes['AC'].mean()
    max_counts_day = max_counts_minutes['AC'].mean()
    min_counts_day = min_counts_minutes['AC'].mean()
    p25_counts_day = p25_counts_minutes['AC'].mean()
    p50_counts_day = p50_counts_minutes['AC'].mean()
    p75_counts_day = p75_counts_minutes['AC'].mean()
    p90_counts_day = p90_counts_minutes['AC'].mean()
    p95_counts_day = p95_counts_minutes['AC'].mean()

    return mean_counts_day, std_counts_day, cv_counts_day, max_counts_day, min_counts_day, p25_counts_day, p50_counts_day, p75_counts_day, p90_counts_day, p95_counts_day, max_counts_day_5min


# feature:
# Angular Features: roll, pitch, and yaw angles
# roll = tan−1(y, z), pitch = tan−1(x, z) and yaw = tan−1(y, x).
# The average (avgroll, avgpitch, avgyaw) and standard deviation (sdroll, sdpitch, sdyaw) of these angles were computed over the window
def angular_features(raw_df):
    roll_counts_second = pd.DataFrame(np.arctan2(raw_df['Y'], raw_df['Z']))
    pitch_counts_second = pd.DataFrame(np.arctan2(raw_df['X'], raw_df['Z']))
    yaw_counts_second = pd.DataFrame(np.arctan2(raw_df['Y'], raw_df['X']))

    roll_counts_second[time_column] = pd.to_datetime(raw_df[time_column])
    pitch_counts_second[time_column] = pd.to_datetime(raw_df[time_column])
    yaw_counts_second[time_column] = pd.to_datetime(raw_df[time_column])

    roll_counts_minute = roll_counts_second.groupby([roll_counts_second[time_column].dt.date, roll_counts_second[time_column].dt.hour, roll_counts_second[time_column].dt.minute])
    pitch_counts_minute = pitch_counts_second.groupby([pitch_counts_second[time_column].dt.date, pitch_counts_second[time_column].dt.hour, pitch_counts_second[time_column].dt.minute])
    yaw_counts_minute = yaw_counts_second.groupby([yaw_counts_second[time_column].dt.date, yaw_counts_second[time_column].dt.hour, yaw_counts_second[time_column].dt.minute])

    roll_counts_minute_mean = roll_counts_minute.mean()
    pitch_counts_minute_mean = pitch_counts_minute.mean()
    yaw_counts_minute_mean = yaw_counts_minute.mean()

    roll_counts_minute_std = roll_counts_minute.std()
    pitch_counts_minute_std = pitch_counts_minute_mean.std()
    yaw_counts_minute_std = yaw_counts_minute_mean.std()

    roll_counts_day_mean = roll_counts_minute_mean.mean().values[0]
    pitch_counts_day_mean = pitch_counts_minute_mean.mean().values[0]
    yaw_counts_day_mean = yaw_counts_minute_mean.mean().values[0]

    roll_counts_day_std = roll_counts_minute_std.mean().values[0]
    pitch_counts_day_std = pitch_counts_minute_std.mean()
    yaw_counts_day_std = yaw_counts_minute_std.mean()

    return roll_counts_day_mean, pitch_counts_day_mean, yaw_counts_day_mean, roll_counts_day_std, pitch_counts_day_std, yaw_counts_day_std

# feature: Fast Fourier Transform (FFT)
def fft_features(raw_df):
    sample_frequency = 30  # 30 hz
    window_size = 60  # 2 seconds
    # fft_frequency = fft.fftfreq(window_size, 1 / sample_frequency)
    half_hz = 1
    five_hz = 10

    amp = np.sqrt(raw_df['X']**2 + raw_df['Y']**2 + raw_df['Z']**2)
    power_spectral_list = []
    i = 0
    while i < len(amp)-60:
        j = i + 60
        fft_results = fft.fft(amp[i:j])[0:int(window_size/2)]
        power = np.abs(fft_results)**2
        power[0] = 0
        max_fre = fft.fftfreq(window_size, 1 / sample_frequency)[np.argmax(power)]
        accumulated_power = np.sum(power[half_hz:five_hz])
        power_spectral_list.append(accumulated_power)
        i = j
    return np.mean(power_spectral_list)

# Iterate over each CSV file to calculate features
final_feature_table = []
c = 0
for file_path in csv_files:
    # Extract the name of the CSV file
    # csv_name = os.path.splitext(os.path.basename(file_path[0]))[0]
    csv_name = os.path.splitext(os.path.basename(file_path))[0]
    print(csv_name)

    # Read the CSV file into a DataFrame
    # df = pd.read_csv(file_path[0])
    df = pd.read_csv(file_path)

    # get counts per second
    counts = get_counts_csv(raw=df, freq=30, epoch=1, verbose=False, time_column=time_column)

    # feature: %active
    counts_minute = active(counts)
    percent_active_day = active_stat(counts_minute)

    # feature: MET
    met_day = met(counts)

    # feature: daily vm
    daily_vm = daily_VM(counts)

    # feature: daily A1
    daily_A1 = np.log(counts['Axis3'].std() + 1)

    # feature: VMI
    vmi = np.log(counts['Axis3'] + 1).std()

    # feature: agg, max, mean, std, ratio of bout time
    aggregate_minutes_each_time, max_minutes_each_time, std_minutes_each_time, mean_minutes_each_time, ratio_minutes_each_time = activity_bout_time(counts_minute)

    # feature: agg, max, mean, std, ratio of bout counts
    aggregate_minutes_each_activity, max_minutes_each_activity, mean_minutes_each_activity, std_minutes_each_activity, ratio_minutes_each_activity = activity_bout_count(counts_minute)

    # feature: basic stats
    mean_counts_day, std_counts_day, cv_counts_day, max_counts_day, min_counts_day, p25_counts_day, p50_counts_day, p75_counts_day, p90_counts_day, p95_counts_day, max_counts_day_5min = basic_stats(counts)

    # feature: angular
    roll_counts_day_mean, pitch_counts_day_mean, yaw_counts_day_mean, roll_counts_day_std, pitch_counts_day_std, yaw_counts_day_std = angular_features(df)

    # feature: Fast Fourier Transform (FFT)
    fft_feature_day = fft_features(df)

    # final feature table
    features = [csv_name, percent_active_day, met_day, daily_vm, daily_A1, vmi,
                aggregate_minutes_each_time[1], aggregate_minutes_each_time[2], aggregate_minutes_each_time[3], aggregate_minutes_each_time[4],
                max_minutes_each_time[1], max_minutes_each_time[2], max_minutes_each_time[3], max_minutes_each_time[4],
                std_minutes_each_time[1], std_minutes_each_time[2], std_minutes_each_time[3], std_minutes_each_time[4],
                mean_minutes_each_time[1], mean_minutes_each_time[2], mean_minutes_each_time[3], mean_minutes_each_time[4],
                ratio_minutes_each_time[1], ratio_minutes_each_time[2], ratio_minutes_each_time[3], ratio_minutes_each_time[4],
                aggregate_minutes_each_activity[1], aggregate_minutes_each_activity[2], aggregate_minutes_each_activity[3], aggregate_minutes_each_activity[4],
                max_minutes_each_activity[1], max_minutes_each_activity[2], max_minutes_each_activity[3], max_minutes_each_activity[4],
                std_minutes_each_activity[1], std_minutes_each_activity[2], std_minutes_each_activity[3], std_minutes_each_activity[4],
                mean_minutes_each_activity[1], mean_minutes_each_activity[2], mean_minutes_each_activity[3], mean_minutes_each_activity[4],
                ratio_minutes_each_activity[1], ratio_minutes_each_activity[2], ratio_minutes_each_activity[3], ratio_minutes_each_activity[4],
                mean_counts_day, std_counts_day, cv_counts_day, max_counts_day, min_counts_day, p25_counts_day, p50_counts_day, p75_counts_day, p90_counts_day, p95_counts_day, max_counts_day_5min,
                roll_counts_day_mean, pitch_counts_day_mean, yaw_counts_day_mean, roll_counts_day_std, pitch_counts_day_std, yaw_counts_day_std,
                fft_feature_day]
    final_feature_table.append(features)

    c += 1
    print(csv_name, c)

final_feature_table = np.array(final_feature_table)

# save to file
csv_file = f'{directory}/feature_extraction/final_feature_table.csv'

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')

    for row in final_feature_table:
        writer.writerow(row)

print("CSV saves successfully:", csv_file)


