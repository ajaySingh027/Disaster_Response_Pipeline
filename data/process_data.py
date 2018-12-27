# import libraries
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
        Reading csv files and creatinf Dataframes
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df



def clean_data(df):
    """
        removing unwanted data columns/rows and cleaning 
    """
    categories = df['categories'].str.split(";", expand=True)
    
    ## select the first row of the categories dataframe
    row = categories.iloc[1]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    print(category_colnames)
    categories.columns = category_colnames

    ##-- Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace(r'\D+', '')
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    # --- # Note: concat is giving int values as decimal. Eg. 1.0 , 0.0 etc. So used merge
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories, left_on = df.index.values, right_on = categories.index.values)
    df.drop(['key_0'], axis=1, inplace=True)

    # drop duplicates
    df= df.drop_duplicates()

     ## - removing 'Related' Col values having '2' as value.
    df['message'][df['related']==2]
    # Remove rows with a related value of 2 from the dataset
    df = df[df['related'] != 2]

    return df



def save_data(df, database_filename):
    """
        saving the dataframe cleaned data in sqlite db
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterTable', engine, index=False)  



def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
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