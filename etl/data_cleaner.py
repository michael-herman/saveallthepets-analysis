#!/usr/bin/env python3

""" Data Cleaner Module
This module contains methods for cleaning data or calls upon scripts containing such methods.
"""


import pandas as pd
import numpy as np
from re import IGNORECASE

# Creating age_years column.
# - First, use the time difference between birthdate and animal_record_create_date.
#   This won't work for every animal, but it's easy.
# - For animals that don't have time difference values:
#    - For animals with known unit types, use the unit and

def create_age_years_col(df):
    """Create an age in years (float) column.

    There are multiple columns that contain information about the age of the animal, often
    in formats that are inconsistent or difficult. This uses multiple approaches to determine
    the age in years.

    If the animal has valid values in the `animal_record_create_date` and `birthdate` columns,
    this is used to create the age. Otherwise, the `age_lbl` column is parsed for date-like
    formats, uses this as a birthdate, and determines the age. After this, the `age_lbl` column
    and `age_units` columns are parsed, cleaned, and an age in years determined.

    Args:
        param1: df. A dataframe containing animals data. Must have the following columns:
            - `animal_record_create_date`
            - `birthdate`
            - `age_lbl`
            - `age_units`

    Returns:
        Adds an `age_years` _np.float_ column and returns the resulting dataframe.

    """

    df.set_index('animal_id')

    # Create blank float col.
    df['age_years'] = np.nan

    # Update based on birthdate and animal_record_create_date column.

    age_years_bdates = df['animal_record_create_date'] - df['birthdate']
    age_years_bdates = age_years_bdates / np.timedelta64(1, 'Y')

    age_years_bdates_float = age_years_bdates.astype(float)
    age_years_bdates_float[age_years_bdates_float <= 0.] = np.nan
    age_years_bdates_float[age_years_bdates_float > 50.] = np.nan

    age_years_bdates_float = age_years_bdates_float.to_frame('age_years') # Convert to data frame w/ proper index/cols.

    df.update(age_years_bdates_float)

    # Update based on date-likes.
    date_likes_pattern = r'(\d{1,4}[/\.\-]\d{1,2}[/\.\-]\d{1,4})'
    date_likes_age_lbl = df['age_lbl'].str.extractall(date_likes_pattern)[0]
    date_likes_age_lbl.reset_index(level='match', drop=True, inplace=True)
    date_likes_age_lbl = pd.to_datetime(date_likes_age_lbl)

    date_likes_df = pd.concat([date_likes_age_lbl, df.loc[:,'animal_record_create_date']], axis=1, join='inner')
    date_likes_df.rename(columns={0:'bdate','animal_record_create_date':'cdate'}, inplace=True)

    date_likes_out = date_likes_df['cdate'] - date_likes_df['bdate']
    date_likes_out = date_likes_out / np.timedelta64(1, 'Y')

    date_likes_out[date_likes_out < 0] = np.nan
    date_likes_out[date_likes_out > 50] = np.nan

    date_likes_out = date_likes_out.to_frame('age_years') # convert to dataframe with proper index/col

    df.update(date_likes_out)


    # Update based on 'year' unit integers.
    year_label_pattern = r'.*((yrs?)|(years?)).*' # True if contains one of 'yr', 'yrs', 'year', 'years'.
    bool_age_lbl_years = df['age_lbl'].str.match(year_label_pattern, na=False, flags=IGNORECASE)
    # bool row indicator wheer age mentioned in age_units.
    bool_age_units_years = df['age_units'].str.match(year_label_pattern, na=False, flags=IGNORECASE)
    # We want to explicitly ignore anything that looks like a date.
    date_likes_pattern = r'(\d{1,4}[/\.\-]\d{1,2}[/\.\-]\d{1,4})'
    bool_date_likes = df['age_lbl'].str.match(date_likes_pattern, na=False)

    year_age_lbl = df.loc[bool_age_lbl_years | bool_age_units_years & ~bool_date_likes, 'age_lbl']
    year_age_lbl = year_age_lbl.str.replace(r'[^\d\.]', '')
    year_age_float = year_age_lbl.astype(float)

    year_age_float[year_age_float <= 0] = np.nan
    year_age_float[year_age_float > 50] = np.nan

    year_age_float = year_age_float.to_frame('age_years')

    df.update(year_age_float)


    # Update based on 'month' unit integers.
    month_label_pattern = r'.*((mos?)|(months?)|(mths?)).*'
    bool_age_lbl_months = df['age_lbl'].str.match(month_label_pattern, na=False, flags=IGNORECASE)
    bool_age_units_months = df['age_units'].str.match(month_label_pattern, na=False, flags=IGNORECASE)
    # We want to explicitly ignore date-likes.
    date_likes_pattern = r'(\d{1,4}[/\.\-]\d{1,2}[/\.\-]\d{1,4})'
    bool_date_likes = df['age_lbl'].str.match(date_likes_pattern, na=False)

    month_age_lbl = df.loc[bool_age_lbl_months | bool_age_units_months & ~bool_date_likes, 'age_lbl']
    month_age_lbl = month_age_lbl.str.replace(r'[^\d\.]', '')
    month_age_float = month_age_lbl.astype(float)
    month_age_float = month_age_float/12

    month_age_float[month_age_float <= 0] = np.nan
    month_age_float[month_age_float > 50] = np.nan

    month_age_float = month_age_float.to_frame('age_years')

    df.update(month_age_float)

    # Update based on 'week' unit integers.
    # 52.1429 weeks in a year.

    week_label_pattern = r'.*((wks?)|(weeks?)).*'
    bool_age_lbl_weeks = df['age_lbl'].str.match(week_label_pattern, na=False, flags=IGNORECASE)
    bool_age_units_weeks = df['age_units'].str.match(week_label_pattern, na=False, flags=IGNORECASE)
    # We want to explicitly ignore date-likes.
    date_likes_pattern = r'(\d{1,4}[/\.\-]\d{1,2}[/\.\-]\d{1,4})'
    bool_date_likes = df['age_lbl'].str.match(date_likes_pattern, na=False)

    week_age_lbl = df.loc[bool_age_lbl_weeks | bool_age_units_weeks & ~bool_date_likes, 'age_lbl']
    week_age_lbl = week_age_lbl.str.replace(r'[^\d\.]', '')
    week_age_float = week_age_lbl.astype(float)
    week_age_float = week_age_float/52.1429 # There are 52.1429 weeks per year.

    week_age_float[week_age_float <= 0] = np.nan
    week_age_float[week_age_float > 50] = np.nan

    week_age_float = week_age_float.to_frame('age_years')

    df.update(week_age_float)

    # Update unknown unit integers.

    bool_no_unit = df['age_units'].isnull()
    bool_has_number = df['age_lbl'].str.match(r'\d+', na=False)

    date_likes_pattern = r'(\d{1,4}[/\.\-]\d{1,2}[/\.\-]\d{1,4})'
    bool_date_likes = df['age_lbl'].str.match(date_likes_pattern, na=False)

    no_unit_age_lbl = df.loc[bool_no_unit &
                             ~bool_date_likes &
                             bool_has_number, 'age_lbl']
    no_unit_age_lbl = no_unit_age_lbl.str.replace(r'[^\d\.]', '')

    no_unit_age_float = no_unit_age_lbl.astype(float)

    no_unit_age_float[no_unit_age_float <= 0] = np.nan
    no_unit_age_float[no_unit_age_float > 50] = np.nan

    no_unit_age_float = no_unit_age_float.to_frame('age_years')

    df.update(no_unit_age_float)

    return df


def clean_species_col(df):
    """ Replace values that aren't "Cat", "Dog", or "Other" with np.nan. Also correct case.

    Given a dataframe of animal data, this will parse the `species` column and replace
    make sure that each value is either "Cat", "Dog", or "Other". It will also correct for
    non-capitalized cases (e.g. "cat" -> "Cat"). It will also trim values that have
    whitespace but are otherwise valid (e.g. " dog" -> "Dog"). Last, it will force anything
    that does not parse as valid to np.nan.

    Args:
        - df: A dataframe of animal data. Must contain a `species` column.

    Returns: The same dataframe with the column parsed and corrected.

    """
    df['species'] = df.loc[:,'species'].str.replace(pat=r'[^(dog)|(cat)|(other)]', repl='', case=False)
    df['species'] = df.loc[:,'species'].replace(to_replace='', value=np.nan)
    
    return df

def clean_gender_col(df):
    """ Replaces values that are not "Male" or "Female" with np.nan. Also corrects for
    capitalization.

    Args:
        - df: A dataframe of animal data. Must contain the column `gender`.

    Returns: The same dataframe with the column parsed and corrected.

    """
    # Replace any characters that is not "Male" or "Female" with blanks.
    # This will trim strings that contain Male or Female. It will also force other strings to ''.
    df['gender'] = df.loc[:,'gender'].str.replace(r'[^(Male)|(Female)]', '')
    # Now replace '' with np.nan.
    df['gender'] = df.loc[:, 'gender'].replace(to_replace='', value=np.nan)
    return df

def clean_weight_col(df):
    """ Checks for non-sensical values in the weight column. Replaces these values with np.nan.

    Non-sensical values are any zero or negative weight values. For cats, any value greater than
    50.0 will be forced to np.nan. (Note: The largest house cat ever was named Meow and he weighed
    a chonky 46.8 lbs.) For dogs, any value greater than 350.0 will be forced to np.nan. (Note: The largest dog ever was named Zorba, a 343 lbs absolute unit of a Mastiff.)

    """
    weight = df.loc[:,'weight']

    # Replace negatives and zeros with np.nan.
    df.loc[df['weight'] <= 0.0, 'weight'] = np.nan

    # Test reasonability for cats. Cats with weight values over 50.0 will be forced to np.nan.
    bool_cat_rows = df['species'].str.match(r'cat', case=False, flags=IGNORECASE)
    bool_bad_cat_weights = bool_cat_rows & df['weight'] > 50.0
    df.loc[bool_bad_cat_weights, 'weight'] = np.nan

    # Test reasonability for dogs. Dogs with weight values over 200.0 will be fonced to np.nan.
    bool_dog_rows = df['species'].str.match(r'dog', case=False, flags=IGNORECASE)
    bool_bad_dog_weights = bool_dog_rows & df['weight'] > 350.0
    df.loc[bool_bad_dog_weights, 'weight'] = np.nan

    return df
