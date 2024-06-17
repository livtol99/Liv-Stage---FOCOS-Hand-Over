# Standard library imports
import csv
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Pool

# Third party imports
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import numpy as np  # Duplicate import, kept only one
import pandas as pd
from joblib import Parallel, delayed
from langdetect import detect, detect_langs, DetectorFactory, LangDetectException
import regex
from scipy.stats import zscore
from unidecode import unidecode
import unicodedata

# Visualization and network analysis libraries
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from adjustText import adjust_text
import networkx as nx

# Local application/library specific imports
import ftfy




"""
This utility file (`utils2.py`) contains a collection of functions used throughout the project for various purposes. 
The functions are organized into distinct sections for ease of navigation and use:

- Data inspection and stats functions

- Data wrangling functions

- Data filtering functions: Functions in this section are used to filter the data based on specific criteria, helping to narrow down the dataset to relevant subsets for analysis.

- Bio processing, language detection, and other text-related functions

- File loading functions

- Plotting functions
"""


# -------------------
# Data inspection and stats functions
# -------------------

def summary_stats(df, print_dtypes=True):
    """
    Prints summary statistics for a DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame for summary statistics.
    print_dtypes (bool, optional): If True, prints column data types. Defaults to True.

    Prints:
    - DataFrame shape, column names, and data types (optional).
    - Unique and duplicate values for 'follower_id', 'id', and 'marker_id' columns (if present).
    - Missing values per column.
    - Total duplicate rows.
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
    df1, df2 (pandas.DataFrame): The DataFrames to compare.
    column (str): The column name to compare.

    Prints:
    - The number of unique values in df1 that don't exist in df2.
    - The number of unique values in df2 that don't exist in df1.
    """
    missing_in_df1 = df1.loc[~df1[column].isin(df2[column]), column]
    missing_in_df2 = df2.loc[~df2[column].isin(df1[column]), column]
    
    print(f"There are {missing_in_df1.nunique()} unique values in df1 that don't exist in df2.")
    print(f"There are {missing_in_df2.nunique()} unique values in df2 that don't exist in df1.")


def calculate_language_percentages(df, column):
    """
    Calculate and print the percentage of each language in a specific column of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    column (str): The column name to analyze.

    Prints:
    - The number and percentage of rows for each language ('fr', 'en', 'unknown', 'NA', and others).
    - The number and percentage of NaN values in the 'description_cleantext' column.
    """
    total_rows = df.shape[0]

    french_rows = df[df[column] == 'fr'].shape[0]
    english_rows = df[df[column] == 'en'].shape[0]
    unknown_rows = df[df[column] == 'unknown'].shape[0]
    NA_rows = df[df[column] == 'NA'].shape[0]
    other_rows = total_rows - french_rows - english_rows - unknown_rows

    french_percent = (french_rows / total_rows) * 100
    english_percent = (english_rows / total_rows) * 100
    unknown_percent = (unknown_rows / total_rows) * 100
    other_percent = 100 - french_percent - english_percent - unknown_percent

    print("French: ", french_rows, "(", french_percent, "%)")
    print("English: ", english_rows, "(", english_percent, "%)")
    print("Unknown: ", unknown_rows, "(", unknown_percent, "%)")
    print("Other: ", other_rows, "(", other_percent, "%)")

    nan_rows = df['description_cleantext'].isna().sum()
    nan_percent = (nan_rows / total_rows) * 100
    print("NaN in description_cleantext: ", nan_rows, "(", nan_percent, "%)")


def calculate_percentage(result, total_rows):
    """
    Calculate the percentage of a part (result) of a total.

    Parameters:
    result (int): The part of the total.
    total_rows (int): The total.

    Returns:
    str: The percentage as a string with a '%' sign.
    """

    percentage = (result / total_rows) * 100
    percentage = round(percentage, 1)  # round to two decimal places
    return str(percentage) + '%'  # add '%' sign

def location_bio_stats(df):
    """
    Calculate and print statistics about location and bio data in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.

    Prints:
    - The number and percentage of unique locations.
    - The number and percentage of users with and without location data.
    - The number and percentage of users with and without bios.
    - The number and percentage of users with both location and bios.
    """
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

# -------------------
# Data wrangling functions
# -------------------

def assign_country(location, city_names):
    """
    Assigns country France based on the presence of french city names within a location string.
    
    Parameters:
    - location (str): The location string to be analyzed.
    - city_names (list or set): A collection of city names used to determine the country. Should be a french city names list.
    
    Returns:
    - str: 'France' if a city name from the list is found in the location, 'Other' otherwise.
    """
    if isinstance(location, str):
        for word in location.split():
            if word in city_names:
                return 'France'
    return 'Other'

def filter_add_jobs_coords(file_number, jobdf):
    """
    Adds coordinates from the CA files to the job title file
    
    Parameters:
    - file_number (int): The file number to construct the file path for coordinates data.
    - jobdf (DataFrame): The DataFrame containing job data to be merged with coordinates.
    
    Returns:
    - DataFrame: The merged DataFrame containing job data with added coordinates.
    
    Performs several steps:
    - Reads coordinates data from a CSV file based on the provided file number.
    - Filters the coordinates data to include only those present in the jobdf DataFrame.
    - Merges the filtered coordinates data with the job data.
    - Strips leading and trailing spaces from the 'title' column.
    - Checks if the output directory exists, creates it if not.
    - Saves the merged DataFrame to a CSV file in the specified directory.
    """
    file_path = f"/home/livtollanes/NewData/coordinates/m{file_number}_coords/m{file_number}_row_coordinates.csv"
    print(f"Used file path: {file_path}") 
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


# -------------------
# Data filtering functions
# -------------------

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

def min_french_followers(df, min_followers):

    """
    Filters a DataFrame to include only rows with a minimum number of French followers and provides information on removed rows.

    Parameters:
    - df (DataFrame): The input DataFrame containing social media data.
    - min_followers (int): The minimum number of French followers required for a row to be included in the filtered DataFrame.

    Returns:
    - tuple: A tuple containing two DataFrames. The first DataFrame includes rows that meet the minimum French followers criterion. 
             The second DataFrame contains information about the rows (brands) that were removed due to not meeting the criterion, 
             including their Twitter name, marker followers, French followers, and type.
    """
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


# -------------------
# Bio processing, language detection, and other text-related functions
# -------------------

def _remove_emoji(string):
    return emoji.demojize(string, delimiters=("<EMOJI:", ">"))

def _remove_emoji_descriptions(string):
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
            bio = _remove_emoji(bio)
            bio = unicodedata.normalize('NFKC', bio)
            bio = _remove_emoji_descriptions(bio)
            bio = regex.sub(r'[^\p{L}\p{N}\p{P}\p{Z}\p{Sc}«»€]', '', bio)
            bio = ''.join(c for c in bio if c <= '\uFFFF')
        else:
            bio = ''
        return bio

    df.loc[:, column + '_cleantext'] = df[column].apply(process_bio)
    return df

def _detect_language(bio):
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
    df['language'] = Parallel(n_jobs=n_jobs)(delayed(_detect_language)(bio) for bio in df[column])
    return df

def process_text(text, stop_words):
    """
    Processes the input text by removing URLs, hashtags, mentions, punctuation, converting to lowercase, and removing stopwords.

    Returns:
    - list: A list of processed words from the input text.
    """
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
    """
    Generates n-grams for a list of words up to a specified size.

    This function creates n-grams for all numbers from 1 up to and including n. 

    Parameters:
    - words (list): A list of words from which to generate n-grams.
    - n (int): The maximum size of the n-gram to generate.

    Returns:
    - list: A list of n-grams generated from the input list of words. Each n-gram is represented as a tuple of words.
    """
    # Create ngrams for all numbers up to and including n
    ngram_list = []
    for i in range(1, n+1):
        ngram_list.extend(list(ngrams(words, i)))
    return ngram_list

def tokenize_bios(df, stop_words):
    """
    Tokenizes bios in a DataFrame and aggregates n-grams for each bio.

    Utilizes `process_text` to clean and tokenize bios by removing URLs, hashtags, mentions, punctuation, 
    converting to lowercase, and removing stopwords. It then generates and aggregates n-grams (for n=1, 2, 3) for each bio using 'get_ngrams'.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing a column 'description_cleantext' with bios.
    - stop_words (set): Set of stopwords to remove during tokenization.

    Returns:
    - pandas.DataFrame: Modified DataFrame with two additional columns:
        - 'description_cleantext_tokens': Tokenized bios with stopwords removed.
        - 'total_n_grams': Aggregated list of all n-grams (for n=1, 2, 3) generated from the tokenized bios.

    The function modifies the input DataFrame in-place by adding the two columns and returns the modified DataFrame. The 'total_n_grams' column is populated by concatenating the lists of n-grams generated for each bio, for n values of 1, 2, and 3.
    """
    # Tokenize the bios
    df['description_cleantext_tokens'] = df['description_cleantext'].apply(lambda x: process_text(x, stop_words))

    # Initialize the total_n_grams column as an empty list
    df['total_n_grams'] = [[] for _ in range(len(df))]

    # Apply the get_ngrams function for n=1, 2, 3 and add the results to total_n_grams
    for n in range(1, 4):
        # Correctly describe the operation as concatenation of lists rather than summing
        df['total_n_grams'] = df.apply(lambda row: row['total_n_grams'] + get_ngrams(row['description_cleantext_tokens'], n), axis=1)
    
    return df

def preprocess_text(text, nlp):
    """
    Preprocesses text by converting to lowercase, removing stopwords and punctuation, and lemmatizing.
    
    Parameters:
    - text (str): The text to preprocess.
    - nlp: A spaCy language model.
    
    Returns:
    - list: A list of lemmatized tokens from the text.
    """
    # Convert the text to lowercase
    text = text.lower()
    
    # Parse the text with spaCy
    doc = nlp(text)
    
    # Remove stopwords and punctuation, and lemmatize the words
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    
    return tokens

def get_ngram_freq(text, n):
    """
    Calculates the frequency of n-grams in a text after preprocessing.
    
    Parameters:
    - text (str): The text to analyze.
    - n (int): The n-gram size.
    
    Returns:
    - Counter: A Counter object with n-grams as keys and their frequencies as values.
    """
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


def separate_ngrams(ngrams):
    """
    Separates n-grams into unigrams, bigrams, and trigrams.
    
    Parameters:
    - ngrams (list): A list of n-grams.
    
    Returns:
    - dict: A dictionary with keys 'unigrams_detected', 'bigrams_detected', 'trigrams_detected' and their corresponding lists of n-grams.
    """
    unigrams = [gram for gram in ngrams if len(gram) == 1 or (len(gram) == 2 and gram[1] == '')]
    bigrams = [gram for gram in ngrams if len(gram) == 2 and gram[1] != '']
    trigrams = [gram for gram in ngrams if len(gram) == 3]
    return {'unigrams_detected': unigrams, 'bigrams_detected': bigrams, 'trigrams_detected': trigrams}

def write_ngrams_to_csv(ngrams, filename):
    """
    Writes n-grams and their counts to a CSV file.

    Parameters:
    - ngrams (list of tuples): A list of tuples where each tuple contains an n-gram (as a tuple of words) and its count.
    - filename (str): The path to the CSV file where the n-grams and counts will be written.

    The function transforms the n-grams into a format suitable for CSV and writes them to the specified file, including a header row.
    """
    # Transform the data into a format suitable for CSV
    ngrams_csv = [(' '.join(ngram), count) for ngram, count in ngrams]

    # Write the data to a CSV file
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Ngram", "Count"])  # write the header
        writer.writerows(ngrams_csv)  # write the data



def find_all_matches2(bio_ngrams, income_df):
    """
    Identifies matches between a list of n-grams and n-grams in a DataFrame, returning associated values.

    This function checks for any overlap between a list of n-grams (bio_ngrams) and n-grams stored in a DataFrame column ('ngrams'). It only considers n-grams longer than one word. For rows in the DataFrame where there's an overlap, it collects values from the 'PCS_ESE' column into a list.

    Parameters:
    - bio_ngrams (list): A list of n-grams to match against the DataFrame.
    - income_df (pandas.DataFrame): A DataFrame with at least two columns: 'ngrams' containing n-grams and 'PCS_ESE' containing values to return for matches.

    Returns:
    - list: A list of 'PCS_ESE' values from the DataFrame where there's an overlap with the bio_ngrams.
    """
    overlap = income_df['ngrams'].apply(lambda x: any(i in x for i in bio_ngrams if len(i.split()) > 1))
    matches = income_df.loc[overlap, 'PCS_ESE'].tolist()
    return matches



# -------------------
# File loading functions
# -------------------
"""
These functions are used to more quickly access the many coordinate files generated from the CA pipeline
"""


def load_all_row_coords_files(n):
    files = []  # list to store all dataframes

    for file_number in range(1, n+1):
        file_path = f"/home/livtollanes/NewData/coordinates/m{file_number}_coords/m{file_number}_row_coordinates.csv"
        print(f"Used file path: {file_path}") 
        df = pd.read_csv(file_path, dtype={'follower_id': str})

        # Add df to list of dataframes
        files.append(df)

    return files

def load_all_column_coords_files(n):
    files = []  # list to store all dataframes

    for file_number in range(1, n+1):
        file_path = f"/home/livtollanes/NewData/coordinates/m{file_number}_coords/m{file_number}_column_coordinates.csv"
        print(f"Used file path: {file_path}") 
        df = pd.read_csv(file_path, dtype={'follower_id': str})

        # Add df to list of dataframes
        files.append(df)

    return files

def load_CA_model_files(n):
    files = []  # list to store all dataframes

    for file_number in range(1, n+1):
        file_path = f"/home/livtollanes/NewData/job_title_coordinates/m{file_number}_jobs_rowcoords.csv"
        print(f"Used file path: {file_path}") 
        df = pd.read_csv(file_path, dtype={'follower_id': str})

        # Replace spaces in column names with underscores
        df.columns = df.columns.str.replace(' ', '_')

        # Add df to list of dataframes
        files.append(df)

    return files


# -------------------
# Plotting functions
# -------------------

def plot_all_brands_together(df, dimension, fontsize= 6):
    """
    Plots all brands together on a scatter plot, colored by type and annotated with Twitter names.

    This function creates a scatter plot for a given DataFrame, where each point represents a brand. Points are colored based on the brand's type and annotated with the brand's Twitter name. The position on the x-axis is determined by a specified dimension, and each brand is assigned a unique y-value to spread them out evenly.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing brand data.
    - dimension (str): The column name in df to use for positioning brands along the x-axis.
    - fontsize (int, optional): The font size for the annotations. Defaults to 6.

    The DataFrame must contain the following columns:
    - 'type2': The type of each brand, which determines the point's color.
    - 'twitter_name': The Twitter handle of each brand, used for annotations.

    A color dictionary within the function maps 'type2' values to specific colors.
    """
    
    # Manually specify a color palette
    color_dict = {'consumption': 'blue', 'information': 'yellowgreen', 'football clubs': 'mediumvioletred', 'education': 'darkorange'}

    # Sort df by the specified dimension values
    df_sorted = df.sort_values(by=dimension)

    # Map 'type2' to colors
    df_sorted['color'] = df_sorted['type2'].map(color_dict)

    # Create a scatter plot
    plt.figure(figsize=(20, 10))

    # Assign each unique twitter_name a unique y-value based on sorted order
    y_values = np.linspace(0, 1, len(df_sorted))

    scatter = plt.scatter(df_sorted[dimension], y_values, c=df_sorted['color'], alpha = 0.6)

    # For each point, add a text label with an arrow
    for i in range(len(df_sorted)):
        twitter_name = df_sorted['twitter_name'].iloc[i]
        xytext = (-30,30) if i % 2 == 0 else (60,-30)  # Alternate label position based on index
        plt.annotate(twitter_name, 
                     (df_sorted[dimension].iloc[i], y_values[i]), 
                     textcoords="offset points", 
                     xytext=xytext, 
                     ha='center', 
                     fontsize=fontsize,  # Set font size here
                     arrowprops=dict(arrowstyle='->', lw=1.5))

    legend_elements = [Patch(facecolor=color, edgecolor=color, label=type2) for type2, color in color_dict.items()]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.yticks([])

    plt.show()

def full_plot_function(df, dimension, types_to_plot, type_style, fontsize=6):
    """
    This function is to be used for plotting all, or some, brands along a dimension of desire.
    Creates a scatter plot of the data in the provided dataframe along the specified dimension. 
    The points are colored according to their type, and a legend is included that shows the color associated with each type.
    Option to plot the old or new_types type classification.
    Option to plot all or some brands only. 

    Parameters:
    df (DataFrame): The input dataframe.
    dimension (str): The column name in the dataframe to be used for the x-axis of the scatter plot.
    types_to_plot (list or str): A list of types to be included in the plot, or 'all' to include all types.
    type_style (str): Determines whether to use the old or new type classification. Expected values are 'old_type' or 'new_type'.
    fontsize (int, optional): Controls the font size of the text labels. Default is 6.
    """
    # Manually specify a color palette for 12 types
    color_dict_old = {'media': 'blue', 'clubs de football': 'yellowgreen', 'sport': 'mediumvioletred', 'grande distribution': 'darkorange',
                      'universities': 'red', 'commerce': 'purple', 'chain restaurants': 'brown', 'luxe vetements et malls': 'pink',
                      'magazine': 'gray', 'party': 'olive', 'ecoles de commerce': 'cyan', 'Lycées professionels': 'magenta'}
    color_dict_new = {'consumption': 'blue', 'information': 'yellowgreen', 'football clubs': 'mediumvioletred', 'education': 'darkorange'}

    # Choose the color dictionary and type column based on type_style
    if type_style == 'old':
        color_dict = color_dict_old
        type_column = 'type'
    elif type_style == 'new':
        color_dict = color_dict_new
        type_column = 'type2'
    else:
        raise ValueError('Invalid type_style. Expected "old_type" or "new_type".')

    # If types_to_plot is 'all', get all unique types from the data
    if types_to_plot == 'all':
        types_to_plot = df[type_column].unique()

    # Sort df by the specified dimension values and filter by type_to_plot
    df_sorted = df[df[type_column].isin(types_to_plot)].sort_values(by=dimension)

    # Map 'type' or 'type2' to colors
    df_sorted['color'] = df_sorted[type_column].map(color_dict)

    # Create a scatter plot
    plt.figure(figsize=(20, 10))

    # Assign each unique twitter_name a unique y-value based on sorted order
    y_values = np.linspace(0, 1, len(df_sorted))

    scatter = plt.scatter(df_sorted[dimension], y_values, c=df_sorted['color'], alpha = 0.6)

    # For each point, add a text label with an arrow
    for i in range(len(df_sorted)):
        twitter_name = df_sorted['twitter_name'].iloc[i]
        # If only one type is being plotted, set xytext to (50, -5)
        xytext = (50, -5) if len(df_sorted['type'].unique()) == 1 else (-30,30) if i % 2 == 0 else (60,-30)
        plt.annotate(twitter_name, 
                     (df_sorted[dimension].iloc[i], y_values[i]), 
                     textcoords="offset points", 
                     xytext=xytext, 
                     ha='center', 
                     fontsize=fontsize,  # Set font size here
                     arrowprops=dict(arrowstyle='->', lw=1.5))

    # Create legend elements based on the unique types in df_sorted
    legend_elements = [Patch(facecolor=color_dict[type2], edgecolor=color_dict[type2], label=type2) for type2 in df_sorted[type_column].unique()]

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.yticks([])

    plt.show()