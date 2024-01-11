import os
import dask.dataframe as dd
import pandas as pd

def load_data(data_path):
    # Define the file paths
    file_paths = {
        'marker_bios': os.path.join(data_path, 'markers_bios_2023-05-19.csv'),
        'marker_followers_bios': os.path.join(data_path, 'markers_followers_bios_2023-05-19.csv'),
        'marker_followers': os.path.join(data_path, 'markers_followers_2023-05-19.csv'),
        'marker_friends': os.path.join(data_path, 'markers_friends_2023-05-19.csv'),
        'readme': os.path.join(data_path, 'README.org')
    }
    # Define the data types for the columns
    dtypes = {'cursor': 'object'}
    # Load the CSV files into Dask DataFrames
    data = {name: dd.read_csv(path, dtype=dtypes, assume_missing=True) for name, path in file_paths.items() if name != 'readme'}
    # Read the 'README.org' file
    with open(file_paths['readme'], 'r') as file:
        data['readme'] = file.read()
    return data

def load_data_in_chunks(data_path, chunksize= 1000):
    # Define the file paths
    file_paths = {
        'bios': os.path.join(data_path, 'markers_bios_2023-05-19.csv'),
        'followers_bios': os.path.join(data_path, 'markers_followers_bios_2023-05-19.csv'),
        'followers': os.path.join(data_path, 'markers_followers_2023-05-19.csv'),
        'friends': os.path.join(data_path, 'markers_friends_2023-05-19.csv'),
        'readme': os.path.join(data_path, 'README.org')
    }
    # Define the data types for the columns
    dtypes = {'cursor': 'object'}
    # Read the CSV files in chunks
    chunks = {name: pd.read_csv(path, dtype=dtypes, chunksize=chunksize, quoting=csv.QUOTE_NONE) for name, path in file_paths.items() if name != 'readme'}
    # Read the 'README.org' file
    with open(file_paths['readme'], 'r') as file:
        chunks['readme'] = file.read()
    return chunks

def count_total_rows(chunks, key):
    total_rows = sum(1 for chunk in chunks[key] for _ in chunk)
    return total_rows