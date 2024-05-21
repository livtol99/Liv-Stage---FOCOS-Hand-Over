
import pandnew_types as pd
from langdetect import detect_langs, LangDetectException, DetectorFactory
from joblib import Parallel, delayed



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



def calculate_language_percentages(df, column):
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

    