# Standard library imports
from collections import defaultdict
import csv
from multiprocessing import Pool
import re

# Third party imports
import emoji
from joblib import Parallel, delayed
from langdetect import detect, detect_langs, DetectorFactory, LangDetectException
import pandas as pd
import regex
from unidecode import unidecode
import unicodedata
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter

import os

# Local application imports
import ftfy


def print_lines(path, file, start_line, end_line):
    """
    Print lines from a file within a given range.
    """
    with open(f"{path}/{file}", 'r') as f:
        print(f"Printing lines from file: {file}")
        for i in range(end_line):
            line = f.readline()
            if i >= start_line:
                print(line)

def fileloader(path, file, req_cols, dtypes):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    path (str): The path to the directory containing the file.
    file (str): The name of the file to load.
    req_cols (list): The list of column names to load from the file.
    dtypes (dict): A dictionary mapping column names to data types.

    Returns:
    pd.DataFrame: The loaded data.
    """
    return pd.read_csv(f"{path}/{file}", delimiter=',', quotechar='"', low_memory=False, usecols=req_cols, dtype=dtypes)

def summary_stats(df, print_dtypes=True):
    """
    Prints summary statistics for a given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame for which to print summary statistics.
    print_dtypes (bool, optional): Whether to print the data types of the DataFrame's columns. Defaults to True.

    The function prints the following:
    - The shape of the DataFrame.
    - The names of the DataFrame's columns.
    - The data types of the DataFrame's columns (if print_dtypes is True).
    - The number of unique and duplicate values in the 'follower_id', 'id', and 'marker_id' columns (if they exist in the DataFrame).
    - The number of missing values in each column.
    - The number of duplicate rows in the DataFrame.
    """
    print("Shape of DataFrame: ", df.shape)
    print("\nColumns in DataFrame: ", df.columns.tolist())
    
    if print_dtypes:
        print("\nData types of columns:\n", df.dtypes)
    
    subset_columns = ['follower_id', 'id', 'marker_id']  

    subset = [col for col in subset_columns if col in df.columns]
    
    for col in subset:
        print(f"\nNumber of unique values in '{col}': ", df[col].nunique())
        duplicates = df[col].duplicated().sum()
        print(f"Number of duplicate values in '{col}': ", duplicates)
    
    print("\nNumber of missing values in each column:")
    for col in df.columns:
        print(f"'{col}': ", df[col].isnull().sum())
    
    print("\nNumber of duplicate rows: ", df.duplicated().sum())

def compare_column_values(df1, df2, column):
    """
    Compare the unique values of a specific column between two pandas DataFrames.

    Parameters:
    column (str): The column name to compare.
    """
    missing_in_df1 = df1.loc[~df1[column].isin(df2[column]), column]
    missing_in_df2 = df2.loc[~df2[column].isin(df1[column]), column]
    
    print(f"There are {missing_in_df1.nunique()} unique values in df1 that don't exist in df2.")
    print(f"There are {missing_in_df2.nunique()} unique values in df2 that don't exist in df1.")


def filter_followers(df, follower_id_column, min_brands):
    """
    Filters a DataFrame to only include followers who are following at least a certain number of brands.

    Parameters:
    df (pandas.DataFrame): The DataFrame to be filtered.
    follower_id_column (str): The name of the column in df that contains the follower IDs.
    min_brands (int): The minimum number of brands a follower must be following to be included in the filtered DataFrame.

    Returns:
    pandas.DataFrame: The filtered DataFrame.
    """
    # Count the number of brands each follower is following
    follower_brand_counts = df.groupby(follower_id_column)['marker_id'].nunique()

    # Get the follower_ids of the followers who are following at least 'min_brands' brands
    valid_followers = follower_brand_counts[follower_brand_counts >= min_brands].index

    # Calculate the number and percentage of followers who follow less than 'min_brands' brands
    invalid_followers_count = len(follower_brand_counts) - len(valid_followers)
    invalid_followers_percentage = (invalid_followers_count / len(follower_brand_counts)) * 100

    print(f"{invalid_followers_count} followers follow less than {min_brands} brands ({invalid_followers_percentage:.2f}% of the total followers).")

    # Filter the DataFrame to only include the valid followers
    filtered_df = df[df[follower_id_column].isin(valid_followers)]

    # Calculate the number and percentage of followers left after the filtering
    valid_followers_count = len(filtered_df[follower_id_column].unique())
    valid_followers_percentage = (valid_followers_count / len(follower_brand_counts)) * 100

    print(f"After removing these followers, {valid_followers_count} followers are left ({valid_followers_percentage:.2f}% of the followers in the inputted df).")
    
    return filtered_df


def streamline_IDs(source, df_tofilter, column):
    """
    Filters a DataFrame based on the presence of values in a specific column of another DataFrame.
    
    Parameters:
    source (DataFrame): The DataFrame to use as the source of values.
    df_tofilter (DataFrame): The DataFrame to filter.
    column (str): The column name to use for filtering.
    
    Returns:
    DataFrame: The filtered DataFrame.
    
    Prints:
    The number of unique values in the specified column of the source DataFrame and the filtered DataFrame.
    The number of rows removed and the number of rows left in the filtered DataFrame.
    """
    initial_rows = len(df_tofilter)
    
    # Filter df_tofilter to only include rows where column value is in source
    df_tofilter_filtered = df_tofilter[df_tofilter[column].isin(source[column])]
    
    final_rows = len(df_tofilter_filtered)

    # Print the number of unique values in each DataFrame
    print(f"Number of unique {column} in source DataFrame: {source[column].nunique()}")
    print(f"Number of unique {column} in filtered DataFrame after filtering: {df_tofilter_filtered[column].nunique()}")
    
    # Print the number of rows removed and left
    print(f"Removed {initial_rows - final_rows} rows from the DataFrame to be filtered.")
    print(f"{final_rows} rows are left in the filtered DataFrame.")

    return df_tofilter_filtered


def filter_by_tweets_and_followers(df, min_followers, min_tweets):
    """
    Filters a DataFrame to only include rows where a follower has a certain minimum number of followers and tweets.
    
    Parameters:
    df (DataFrame): The DataFrame to filter.
    min_followers (int): The minimum number of followers a follower must have.
    min_tweets (int): The minimum number of tweets a follower must have.
    
    Returns:
    DataFrame: The filtered DataFrame.
    
    Prints:
    The number of rows removed and the number of rows left in the DataFrame.
    """
    initial_rows = len(df)
    
    # Filter df to only include rows where a follower has min_followers or more followers and min_tweets or more tweets
    df_filtered = df[(df['followers'] >= min_followers) & (df['tweets'] >= min_tweets)]
    
    final_rows = len(df_filtered)

    # Print the number of rows removed and left
    print(f"Removed {initial_rows - final_rows} rows.")
    print(f"{final_rows} rows are left.")

    return df_filtered


def print_lines(path, file, start_line, end_line):
    """
    Print lines from a file within a given range.
    """
    with open(f"{path}/{file}", 'r') as f:
        print(f"Printing lines from file: {file}")
        for i in range(end_line):
            line = f.readline()
            if i >= start_line:
                print(line)



def remove_emoji(string):
    return emoji.demojize(string, delimiters=("<EMOJI:", ">"))

def remove_emoji_descriptions(string):
    return re.sub(r'<EMOJI:.*?>', '', string)




def process_description(df, column):
    """
    Process a column of a DataFrame.
    Removes emojis, special characters and unusual fonts, fixes text issues while preserving accents.

    Parameters:
    df (DataFrame): The DataFrame to process.
    column (str): The name of the column to process.

    Returns:
    DataFrame: The processed DataFrame.
    """
    df = df.copy()
    def process_bio(bio):
        if pd.notnull(bio):
            bio = remove_emoji(bio)
            bio = unicodedata.normalize('NFKC', bio)
            bio = remove_emoji_descriptions(bio)
            bio = regex.sub(r'[^\p{L}\p{N}\p{P}\p{Z}\p{Sc}«»€]', '', bio)
            bio = ''.join(c for c in bio if c <= '\uFFFF')
        else:
            bio = ''
        return bio

    df.loc[:, column + '_cleantext'] = df[column].apply(process_bio)
    return df



def detect_language(bio):
    """
    Detect the language of a string using the langdetect library.

    Parameters:
    bio (str): The string to process.

    Returns:
    str: The language of the string, or 'unknown' if the language could not be detected or if the input is not a string.
    """
    if pd.isna(bio) or bio.strip() == '':
        return 'unknown'
    try:
        detected_languages = detect_langs(bio)
        # The first language in the list is the most probable
        most_probable_language = detected_languages[0]
        return str(most_probable_language.lang)
    except LangDetectException:
        return 'unknown'

def add_and_detect_language(df, column, seed=3, n_jobs=-1):

    """
    Add a language column to a DataFrame and detect the language for each row.

    Parameters:
    df (DataFrame): The DataFrame to process.
    column (str): The column to detect language from.
    seed (int): The seed for the language detection algorithm.
    n_jobs (int): The number of CPU cores to use. -1 means using all processors.

    Returns:
    DataFrame: The DataFrame with the added language column.
    """
    DetectorFactory.seed = seed
    df['language'] = Parallel(n_jobs=n_jobs)(delayed(detect_language)(bio) for bio in df[column])
    return df

# def get_language(text):
#     if pd.isnull(text):
#         return 'unknown'
#     identifier = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=200)
#     result = identifier.FindLanguage(text)
#     return result.language

# def detect_language_gcld3(df, column, n_jobs=-1):
#     if column not in df.columns:
#         return df

#     df['language'] = Parallel(n_jobs=n_jobs)(delayed(get_language)(text) for text in df[column])

#     return df


def split_by_language(df, language):
    """
    Split a DataFrame by language.

    Parameters:
    df (DataFrame): The DataFrame to split.
    language (str): The language to split by.

    Returns:
    DataFrame, DataFrame: The DataFrame with the specified language, and the DataFrame with all other languages.
    """
    df_language = df[df['language'] == language]
    df_other = df[df['language'] != language]
    return df_language, df_other


def calculate_percentage(result, total_rows):
    percentage = (result / total_rows) * 100
    percentage = round(percentage, 1)  # round to two decimal places
    return str(percentage) + '%'  # add '%' sign

# Define a function to assign country
def assign_country(location, city_names):
    if isinstance(location, str):
        for word in location.split():
            if word in city_names:
                return 'France'
    return 'Other'

def location_bio_stats(df):
    total_rows = len(df)
    
    # Define a helper function to calculate and print a statistic
    def print_stat(name, count):
        percentage = calculate_percentage(count, total_rows)
        print(f'{name}: {count} ({percentage})')
    
    # Calculate and print each statistic
    print_stat('Unique locations', df['location'].nunique())
    print_stat('Users with location data', df['location'].notna().sum())
    print_stat('Users without location data', df['location'].isna().sum())
    print_stat('Users with bios', df['description_cleantext'].notna().sum())
    print_stat('Users without bios', df['description_cleantext'].isna().sum())
    print_stat('Users with both location and bios', df[(df['location'].notna()) & (df['description_cleantext'].notna())].shape[0])

def min_french_followers(df, min_followers):
    # Filter rows with 'french_followers' less than min_followers
    filtered_df = df[df['french_followers'] >= min_followers]

    # Find the rows that were removed
    removed_rows = df.loc[~df.index.isin(filtered_df.index)]

    # Get the 'twitter_name' and 'french_followers' columns of the removed rows
    removed_info = removed_rows[['twitter_name', 'marker_followers','french_followers', 'type']]

    # Remove duplicate 'twitter_name' rows
    removed_info = removed_info.drop_duplicates(subset='twitter_name')

    # Print the total number of brands removed
    print(f"Total brands removed: {removed_info['twitter_name'].nunique()}")

    return filtered_df, removed_info


def check_types(df, group_column, count_column):
    # Check if the group column exists in the DataFrame
    if group_column in df.columns:
        # Calculate the number of unique values in the group column
        unique_values = df[group_column].nunique()
        print(f"Number of unique values in '{group_column}': {unique_values}\n")

        # Group by the group column and calculate the min and max of the count column
        group = df.groupby(group_column)[count_column].agg(['min', 'max'])

        # Print the min and max for each group
        for index, row in group.iterrows():
            print(f"{group_column} = {index}:\n"
                  f"  Min '{count_column}': {row['min']}\n"
                  f"  Max '{count_column}': {row['max']}\n")
    else:
        print(f"'{group_column}' does not exist in the DataFrame.")


def add_label(df_to_change, df_with_labels, label_column):
    changed_df = df_to_change.merge(df_with_labels[['twitter_name', label_column]], on='twitter_name', how='left')
    return changed_df

def count_unique_labels(df):
    df_unique = df.drop_duplicates(subset=['twitter_name', 'label'])
    label_counts = df_unique['label'].value_counts()
    return label_counts


def create_marker_projection(df, unit_normal, dimensions):
    """
    This function creates a projection of the data onto a hyperplane defined by a unit normal.

    Parameters:
    df (DataFrame): The input dataframe.
    unit_normal (list): The coordinates for the hyperplane unit normal.
    dimensions (list): The column names in the dataframe to be used for the dimensions of the projection.

    Returns:
    DataFrame: The dataframe with the added projection.
    """
    # Make the unit normal into an array to facilitate operations
    unit_normal = np.array(unit_normal)

    # Select the columns corresponding to the dimensions
    marker_coordinates = df[dimensions].values

    # Perform the dot product operation between the coordinates and the hyperplane unit normal (our new dimension)
    projections = np.dot(marker_coordinates, unit_normal)

    # Add the projections to your DataFrame
    df['projection'] = projections

    return df

    return df

def create_user_projection(df, unit_normal, dimensions):
    """
    This function creates a projection of the data onto a hyperplane defined by a unit normal.
    It adds 'follower_id' instead of 'twitter_name'.

    Parameters:
    df (DataFrame): The input dataframe.
    unit_normal (list): The coordinates for the hyperplane unit normal.
    dimensions (list): The column names in the dataframe to be used for the dimensions of the projection.

    Returns:
    DataFrame: The dataframe with the added projection and 'follower_id'.
    """
    # Make the unit normal into an array to facilitate operations
    unit_normal = np.array(unit_normal)

    # Select the columns corresponding to the dimensions
    user_coordinates = df[dimensions].values

    # Perform the dot product operation between the coordinates and the hyperplane unit normal (our new dimension)
    projections = np.dot(user_coordinates, unit_normal)

    # Add the projections to your DataFrame
    df['projection'] = projections

    # Reorder the DataFrame based on the 'projection' column from highest to lowest
    df = df.sort_values(by='projection', ascending=False)

    return df


def write_ngrams_to_csv(ngrams, filename):
    # Transform the data into a format suitable for CSV
    ngrams_csv = [(' '.join(ngram), count) for ngram, count in ngrams]

    # Write the data to a CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Ngram", "Count"])  # write the header
        writer.writerows(ngrams_csv)  # write the data

def get_ngram_freq(text, n):
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Tokenize the string into words
    words = word_tokenize(text)

    # Remove punctuation and convert to lower case
    words = [word.lower() for word in words if word.isalpha()]

    # Remove French stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('french'))
    words = [word for word in words if word not in stop_words]

    # Create ngrams
    ngram = ngrams(words, n)
    ngram_freq = Counter(ngram)

    return ngram_freq


def process_text(text, stop_words):
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Replace hashtags and mentions with just the word
    text = re.sub(r'[@#](\w+)', r'\1', text)
    # Tokenize the string into words
    words = nltk.word_tokenize(text)
    # Remove punctuation and convert to lower case
    words = [word.lower() for word in words if word.isalpha()]
    # Remove French stopwords
    words = [word for word in words if word not in stop_words]
    return words

def get_ngrams(words, n):
    # Create ngrams for all numbers up to and including n
    ngram_list = []
    for i in range(1, n+1):
        ngram_list.extend(list(ngrams(words, i)))
    return ngram_list

def separate_ngrams(ngrams):
    unigrams = [gram for gram in ngrams if len(gram) == 1 or (len(gram) == 2 and gram[1] == '')]
    bigrams = [gram for gram in ngrams if len(gram) == 2 and gram[1] != '']
    trigrams = [gram for gram in ngrams if len(gram) == 3]
    return {'unigrams_detected': unigrams, 'bigrams_detected': bigrams, 'trigrams_detected': trigrams}

def tokenize_bios(df, stop_words):
    # Tokenize the bios
    df['description_cleantext_tokens'] = df['description_cleantext'].apply(lambda x: process_text(x, stop_words))

    # Initialize the total_n_grams column as an empty list
    df['total_n_grams'] = [[] for _ in range(len(df))]

    # Apply the get_ngrams function for n=1, 2, 3 and add the results to total_n_grams
    for n in range(1, 4):
        df['total_n_grams'] = df['total_n_grams'] + df['description_cleantext_tokens'].apply(lambda x: get_ngrams(x, n))
    
    return df

def calculate_common_ngrams(set1, set2):
    return len(set1.intersection(set2))


def find_all_matches2(bio_ngrams, income_df):
    overlap = income_df['ngrams'].apply(lambda x: any(i in x for i in bio_ngrams if len(i.split()) > 1))
    matches = income_df.loc[overlap, 'PCS_ESE'].tolist()
    return matches

def preprocess_text(text, nlp):
    # Convert the text to lowercase
    text = text.lower()
    
    # Parse the text with spaCy
    doc = nlp(text)
    
    # Remove stopwords and punctuation, and lemmatize the words
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    return tokens


def filter_add_jobs_coords(file_number, jobdf):
    file_path = f"/home/livtollanes/NewData/coordinates/m{file_number}_coords/m{file_number}_row_coordinates.csv"
    print(f"Constructed file path: {file_path}") 
    df = pd.read_csv(file_path, sep = ',', dtype={'follower_id': str})

    # Filter df based on jobdf
    comparison_ids = jobdf['follower_id'].unique()
    df = df[df['follower_id'].isin(comparison_ids)]

    # Merge
    jobdf = jobdf.drop(columns=['0'])
    df = pd.merge(df, jobdf, on='follower_id', how='left')

    #Strip leading and trailing spaces from 'title' column
    df['title'] = df['title'].str.strip()

    # Check if directory exists, if not, create it
    directory = "/home/livtollanes/NewData/job_title_coordinates"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Constructed job coord path: {directory}")
    
    # Save df to CSV file in directory
    output_file_path = f"{directory}/m{file_number}_jobs_rowcoords.csv"
    df.to_csv(output_file_path, sep = ',', index = False)

    return df


def load_all_row_coords_files(n):
    files = []  # list to store all dataframes

    for file_number in range(1, n+1):
        file_path = f"/home/livtollanes/NewData/coordinates/m{file_number}_coords/m{file_number}_row_coordinates.csv"
        print(f"Used file path: {file_path}") 
        df = pd.read_csv(file_path, dtype={'follower_id': str})

        # Add df to list of dataframes
        files.append(df)

    return files

def load_CA_model_files(file_number):
    file_path = f"/home/livtollanes/NewData/job_title_coordinates/m{file_number}_jobs_rowcoords.csv"
    print(f"Used file path: {file_path}") 
    df = pd.read_csv(file_path, dtype={'follower_id': str})

    return df