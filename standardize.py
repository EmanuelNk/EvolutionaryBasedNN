import csv
from os import replace

import pandas as pd
from pandas import Series, DataFrame
import os
from statistics import stdev, mean
SAMPLES_FILE_LENGTH = 10000

means = [3618.046230952381,
         99.97989743800925,
         3778.9476726666667,
         100.01718145482234]

stdevs = [13129.675504774228,
          0.159076714304619,
          16734.941204971783,
          0.15862270720764318]

# means = [0.28106912296870923,
#          0.5060018012914865,
#          0.2804429870575669,
#          0.5052074796360044]

# stdevs = [0.25897761046657164,
#           0.3009245165264731,
#           0.2592936059128438,
#           0.3009647530591276]


def apply_std(x: Series):
    if x.name == 'label':
        return x
    name_idx = x.name
    name_idx = int(name_idx.replace('IN', ''))
    mean = means[name_idx % 4]
    std_dev = stdevs[name_idx % 4]
    return x.apply(lambda v: (v - mean) / std_dev).round(4)


def calc_mean_stdev(df: DataFrame):
    means = []
    stdevs = []
    for i in range(1, 5):
        flat_data = pd.Series(df[df.columns[i::4]].values.ravel('F'))
        mean = flat_data.mean()
        stdev = flat_data.std()
        means.append(mean)
        stdevs.append(stdev)
    return means, stdevs


def standardize(df):
    df_std = df.apply(apply_std)
    return df_std


def dec_samples(df):
    return df.groupby(['label']).head(SAMPLES_FILE_LENGTH)


def preprocess(input_path: str, output_path: str):
    if not os.path.isfile(output_path):
        df = pd.read_csv(input_path, names=['label'] + list(range(120)))
        print('normalize rows...')
        # norm_df = normalize_rows(df)
        # print('calc mean stdev')
        # calc_mean_stdev(norm_df)
        standardize(normalize_rows(dec_samples(df))).to_csv(output_path, header=False, index=False)


def preprocess_rami(input_path: str, output_path: str):
    new_file = []
    with open(input_path, 'r') as input_file:
        reader = csv.reader(input_file)
        for individual in reader:
            new_individual = [individual[0]]
            for i in range(1, 5):
                sensor_data = list(map(float, individual[i::4]))
                sensor_diff = [sensor_data[j + 1] - sensor_data[j] for j in range(0, len(sensor_data)-1)]
                new_individual.append(mean(sensor_data))
                new_individual.append(stdev(sensor_data))
                new_individual.append(mean(sensor_diff))
                new_individual.append(stdev(sensor_diff))
            new_file.append(new_individual)
    with open(output_path, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(new_file)


def preprocess_rami_3_mean_diff(input_path: str, output_path: str):
    new_file = []
    with open(input_path, 'r') as input_file:
        reader = csv.reader(input_file)
        for individual in reader:
            new_individual = [individual[0]]
            for i in range(1, 5):
                sensor_data = list(map(float, individual[i::4]))
                sensor_diff = [sensor_data[j + 1] - sensor_data[j] for j in range(0, len(sensor_data)-1)]
                sensor_data_1 = mean(sensor_data[0:10])
                sensor_data_2 = mean(sensor_data[10:20])
                sensor_data_3 = mean(sensor_data[20:30])
                new_individual.append(sensor_data_3-sensor_data_2)
                new_individual.append(sensor_data_2-sensor_data_1)
                new_individual.append(mean(sensor_data))
                new_individual.append(stdev(sensor_data))
                new_individual.append(mean(sensor_diff))
                new_individual.append(stdev(sensor_diff))
            new_file.append(new_individual)
    with open(output_path, 'w') as output_file:
        writer = csv.writer(output_file)
        writer.writerows(new_file)

def minmax_norm(x):
    min = x.min()
    max = x.max()
    if min == max:
        return 0
    return (x - min) / (max - min)


def normalize_rows(df):
    labels_df = df[['label']]
    for i in range(1, 5):
        data_df = df[list(df.columns[i::4])]
        labels_df = labels_df.join(data_df.apply(minmax_norm, axis=1))
    return labels_df.reindex(['label'] + ['IN' + str(i) for i in range(120)], axis=1)


def split_dataset(input_path: str, output_path: str):
    df = pd.read_csv(input_path, names=['label'] + list(range(120)))
    output_path = '.'.join(output_path.split('.')[:-1])
    for i in range(1, 5):
        new_df = df[['label'] + list(df.columns[i::4])]
        new_df.to_csv(output_path + f'_{i}.csv', header=False, index=False)


def generate_input_files(input_path: str):
    properties = [f'IN{i}' for i in range(120)]
    file_name, file_ext = os.path.splitext(os.path.basename(input_path))
    df = pd.read_csv(input_path, names=['label'] + properties)
    print(f'standartizing {file_name}...')
    standardize(df).to_csv(f'{file_name}_std{file_ext}', header=False, index=False)
    print(f'preprocessing {file_name}...')
    preprocess_rami(f'{file_name}_std{file_ext}', f'{file_name}_std_processed{file_ext}')
    return f'{file_name}_std_processed{file_ext}'


if __name__ == '__main__':
    preprocess(os.path.join('data', "train.csv"))
