import sys
import requests
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load disaster response csv datasets and save as merged pandas dataframe
    :param messages_filepath: filepath to csv with messages
    :param categories_filepath: filepath to csv with message categories
    :return: pandas dataframe
    '''
    # load csv datasets
    df_mess = pd.read_csv(messages_filepath)
    df_cat = pd.read_csv(categories_filepath)

    # merge raw dataframes on id
    df = pd.merge(df_mess, df_cat, on='id')

    return df


def clean_data(df):
    '''
    cleans the raw data in given pandas dataframe
    :param df: pandas dataframe
    :return: pandas dataframe with categorical columns and duplicates removed
    '''
    ### get categories as categorical columns ###
    # get separate column for each category in df
    categories = df['categories'].str.split(";", expand=True)

    # rename columns based on first row
    row = categories.iloc[0]
    category_colnames = list(row.apply(lambda x: x.split('-')[0]))
    categories.columns = category_colnames

    # convert to categorical columns with 0/1 values
    for column in categories:
        categories[column] = categories[column].str.strip().str[-1].astype('int')

    # drop original categories column and append new categorical columns to dataframe
    df = pd.concat([df.drop(['categories'], axis=1), categories], axis=1)

    ### drop duplicates ###
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filepath):
    '''
    loads pandas dataframe into SQLite database
    :param df: pandas dataframe
    :param database_filepath: filepath for the SQLite database to upload to
    :return: None
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterResponseMessageData', engine, index=False, if_exists='replace')
    return


def main():
    '''
    runs the ETL pipeline (loads and cleans data, saves it into SQLite database - to execute, run
    'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
    from the command line
    :return: None
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()