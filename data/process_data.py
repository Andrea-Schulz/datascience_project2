###################
# ETL Pipeline - to run data processing on disaster response message data, execute:
# 'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
###################

import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load disaster response datasets and save as pandas dataframe
    :param messages_filepath (str): filepath to .csv file with messages
    :param categories_filepath (str): filepath to .csv file with message categories
    :return: df (pandas dataframe): dataframe with merged datasets
    '''
    # load csv datasets
    df_mess = pd.read_csv(messages_filepath)
    df_cat = pd.read_csv(categories_filepath)

    # merge raw dataframes on id
    df = pd.merge(df_mess, df_cat, on='id')

    return df


def clean_data(df):
    '''
    cleans the input data
    :param df (pandas dataframe): dataframe with uncleaned data
    :return: df (pandas dataframe): dataframe with categorical columns and duplicates removed
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

    # drop original categories column and concat new categorical columns to dataframe
    df = pd.concat([df.drop(['categories'], axis=1), categories], axis=1)

    ### drop duplicates ###
    df.drop_duplicates(inplace=True)

    # remove rows where "related" is categorized as '2' (and '2' only): often the messages have no proper translation,
    # so fitting them to the desired 0/1 encoding would likely distort the dataset
    df = df[df['related'] != 2]

    return df


def save_data(df, database_filepath):
    '''
    saves pandas dataframe as a table in a SQLite database
    :param df (pandas dataframe): dataframe to be saved and loaded into database
    :param database_filepath (str): filepath for the SQLite database to upload to
    :return: None
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('DisasterResponseMessageData', engine, index=False, if_exists='replace')
    return


def main():
    '''
    run ETL pipeline (load and clean data, save it into SQLite database)
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