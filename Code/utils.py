import pandas as pd
from unidecode import unidecode
from langdetect import detect, DetectorFactory, LangDetectException
import emoji
from collections import defaultdict
import csv


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


def add_extrabrands(indices_to_change, dfm_other, dfm_french, path):
    """
    Process dataframes based on given indices, print the head of both dataframes, and save them to CSV files.

    Parameters:
    indices_to_change (list): The indices to change in dfm_other.
    dfm_other (DataFrame): The dataframe to process.
    dfm_french (DataFrame): The dataframe to add selected rows to.
    path (str): The path to save the CSV files.

    Prints:
    The head of both dataframes and a success message after saving the dataframes to CSV files.
    """
    dfm_other = dfm_other.copy()
    dfm_other['corrected_language_country'] = ''
    dfm_other.loc[indices_to_change, 'corrected_language_country'] = 'fr'

    # Adding the identified rows to the french df
    selected_rows = dfm_other[dfm_other['corrected_language_country'] == 'fr']
    dfm_french = pd.concat([dfm_french, selected_rows])

    # Print the head of both dataframes
    print("dfm_french:")
    print(dfm_french.head())
    print("\ndfm_other:")
    print(dfm_other.head())

    # Write the dataframes to CSV files
    dfm_french.to_csv(f'{path}/french_brands.csv', index=False)
    dfm_other.to_csv(f'{path}/other_brands.csv', index=False)

    print("Both dataframes were successfully written to CSV files.")



def create_brands_per_follower_dict(file_path):
    """
    Create a dictionary of follower_id as keys and brands as values from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    tuple: A tuple containing two dictionaries. The first dictionary has follower_id as keys and brands as values.
           The second dictionary has follower_id as keys and the count of brands as values.
    """
    # Create a dictionary of keys:follower_id and value: brands
    brands_per_follower = defaultdict(set)

    # Open the CSV file
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Add the brand id to the set of brands for this follower
            brands_per_follower[row['follower_id']].add(row['id'])

    # Convert the sets to counts to see how many brands each follower follows
    brands_per_follower_count = {follower_id: len(brands) for follower_id, brands in brands_per_follower.items()}

    return brands_per_follower, brands_per_follower_count

def create_followers_per_brand_dict(df):
    """
    Create a dictionary of brand_id as keys and followers as values from a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame.

    Returns:
    dict: A dictionary with brand_id as keys and the count of followers as values.
    """
    # Create a dictionary of keys:brand_id and value: followers
    followers_per_brand = defaultdict(set)

    for index, row in df.iterrows():
        # Add the follower id to the set of followers for this brand
        followers_per_brand[row['id']].add(row['follower_id'])

    # Convert the sets to counts to see how many followers each brand has
    followers_per_brand_count = {brand_id: len(followers) for brand_id, followers in followers_per_brand.items()}

    return followers_per_brand_count


def inspect_and_filter_followers(brands_per_follower, num_brands, num_items, remove=False):
    """
    Filter followers who follow less than a certain number of brands, print statistics, and optionally remove them.

    Parameters:
    brands_per_follower (dict): The dictionary of followers and brands.
    num_brands (int): The minimum number of brands a follower should follow.
    remove (bool): Whether to remove followers who follow less than num_brands brands.

    Returns:
    dict: The filtered dictionary of followers and brands if remove is True, otherwise the original dictionary.
    """
    # Initialize filtered_brands_per_follower with the original dictionary
    filtered_brands_per_follower = brands_per_follower

    # How many users follow less than num_brands brands?
    count = sum(1 for brands in brands_per_follower.values() if len(brands) < num_brands)

    # Calculate the percentage
    percentage = (count / len(brands_per_follower)) * 100

    # Calculate the percentage of the remaining data
    remaining_percentage = 100 - percentage

    # Calculate the number of remaining users
    remaining_users = len(brands_per_follower) - count

    print(f"The number of keys that follow less than {num_brands} IDs is {count}, which is {percentage:.2f}% of the total. Removing these leaves {remaining_percentage:.2f}% of the data, or {remaining_users} users.")

    if remove:
        print("Deleting rows ...")
        # Filter the dictionary
        filtered_brands_per_follower = {follower_id: brands for follower_id, brands in brands_per_follower.items() if len(brands) >= num_brands}

        # Inspect first key value pairs in filtered dict
        items = iter(filtered_brands_per_follower.items())

        # Get the first 5 items
        print(f"First {num_items} items in the filtered dictionary:")
        for _ in range(num_items):
            print(next(items))

    return filtered_brands_per_follower




