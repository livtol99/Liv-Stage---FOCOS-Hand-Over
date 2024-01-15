import pandas as pd
from unidecode import unidecode
from langdetect import detect, DetectorFactory, LangDetectException
import emoji


def print_lines(path, file, start_line, end_line):
    """
    Print lines from a file within a given range.
    """
    with open(f"{path}/{file}", 'r') as f:
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

def remove_emoji(string):
    """
    Remove emojis from a string.

    Parameters:
    string (str): The string to process.

    Returns:
    str: The string without emojis.
    """
    return emoji.demojize(string, delimiters=("", ""))

def convert_to_regular_script(string):
    """
    Convert a string to regular script.

    Parameters:
    string (str): The string to process.

    Returns:
    str: The string in regular script.
    """
    return unidecode(string)

def detect_language(bio):
    """
    Detect the language of a string.

    Parameters:
    bio (str): The string to process.

    Returns:
    str: The language of the string, or 'unknown' if the language could not be detected.
    """
    DetectorFactory.seed = 3
    try:
        return detect(bio)
    except LangDetectException:
        return 'unknown'

def process_description(df):
    """
    Process the 'description' column of a DataFrame.
    Removes emojies, converts to regular script, and detects the language.

    Parameters:
    df (DataFrame): The DataFrame to process.

    Returns:
    DataFrame: The processed DataFrame.
    """
    df['description_noems'] = df['description'].apply(lambda bio: remove_emoji(bio) if pd.notnull(bio) else '')
    df['description_noems'] = df['description_noems'].apply(lambda bio: convert_to_regular_script(bio) if pd.notnull(bio) else '')
    df['language'] = df['description_noems'].apply(lambda bio: detect_language(bio) if bio.strip() != '' else 'unknown')
    return df

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

def print_df_info(df, name):
    """
    Print the number of rows in a DataFrame.

    Parameters:
    df (DataFrame): The DataFrame to print information about.
    name (str): The name of the DataFrame.
    """
    print(f"Number of rows in {name}: {df.shape[0]}")


    
def inspect_dict(dictionary, n):
    """
    Count the number of unique values and keys in a dictionary, and print the first n items.

    Parameters:
    dictionary (dict): The dictionary to analyze.
    n (int): The number of items to print.

    Prints:
    The number of unique values and keys in the dictionary, and the first n items.
    """
    # Count unique values (brands) in the dictionary
    unique_values = set(value for values in dictionary.values() for value in values)
    num_unique_values = len(unique_values)

    # Count keys (follower_id) in the dictionary
    num_keys = len(dictionary)

    print(f"The number of unique values in the dictionary is {num_unique_values}.")
    print(f"The number of keys in the dictionary is {num_keys}.")

    # Get an iterator over the dictionary's items
    items = iter(dictionary.items())

    # Get the first n items
    print(f"First {n} items in the dictionary:")
    for _ in range(n):
        print(next(items))

