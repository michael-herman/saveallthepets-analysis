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
    """ Add a `age_years` column (float) specifying the age in years.
    Age information input is inconsistent, sometimes using months, sometimes years, et.
    This parses the `age_units`, `age_lbl`, `birthdate` and `animal_record_create_date` fields
    to determine the animal's age.
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
