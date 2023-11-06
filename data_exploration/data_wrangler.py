'''
Imports all the modules needed by the class.
'''
import multiprocessing as mp
import pandas as pd
import numpy as np
import config.config as cf
class DataWrangler(object):
    """
    A simple DataWrangler class.
    Attributes:
        file_name (str): The name of the file bearing our dataset to be wrangled.
    Methods:
        load_data(): Loads the dataset from the file into a pandas dataframe.
        shape_of_data(df_data): Returns the shape of the dataframe passed to it.
        summary_statistics(column_data): Returns a summary statistics of pandas dataframe 
        column passed to it.
        find_missing_values(df_data): Returns a list of missing values (nan) by columns.
        duplicated_rows(df_data): Returns duplicated rows
        drop_duplicates(df_data): Returns a pandas dataframe with duplicated rows dropped.
        fill_missing_values(df_data): Fills missing values (nan) in the dataframe with the mean 
        value of each column
        find_outliers(df_column): Returns all values that are outliers in the numeric 
        column passed to it.
        is_outlier(value): Returns the value if it is an outlier
        fill_outliers_with_mean(outliers, df_column): fills all entries in the dataframe 
        column passed to it, that are outliers with the mean value of the column.
        sort_data(df_data): Sorts the dataframe by the 'x' column    

        Example:
            data_wrangler = DataWrangler(file_name="train.csv")
            df_data = data_wrangler.load_data()
            print(df_data.head(5))  # Output: The first five rows of the train dataset
    """
def __init__(self, file_name):
    '''
    Constructor for the DataWrangler Class
    file_name: The Name of the file from which we get our raw data for wrangling
    '''
    self.file_name = file_name
    self.lower_bound_outlier = 0
    self.upper_bound_outlier = 0
def load_data(self):
    '''
    Loads the file received by constructor to pandas dataframe
    return: A pandas dataframe bearing the content of our file passed to constructor
    '''
    df_data = pd.read_csv(filepath_or_buffer=cf.INPUT_FILE_PATH+self.file_name, 
                          sep=",", encoding="latin1")
    return df_data
def shape_of_data(df_data):
    '''
    Computes the shape of the dataframe passed to it
    return: A tuple, which is the shape of the dataframe.
    '''
    df_shape = df_data.shape
    return df_shape
def summary_statistics(column_data):
    '''
    Computes the summary statistic of each column of the dataframe
    return: A dataframe, which gives a summary statistic of each
    column in the dataframe.
    '''
    return column_data.describe()
def find_missing_values(df_data):
    '''
    Computes the count of missing values in each column of the dataframe
    return: Returns a pandas series bearing the details on the count of missing
    values in each column of the dataframe
    '''
    df_result = df_data.apply(lambda x: sum(x.isnull()), axis=0)
    return df_result
def duplicated_rows(df_data):
    '''
    Finds all duplicated rows in the dataframe
    return: Returns a dataframe bearing duplicated rows in the dataframe
    '''
    df_duplicated = df_data.duplicated()
    return df_duplicated
def drop_duplicated(df_data):
    '''
    Drops all duplicated rows in the dataframe and keeps the first
    occurrence
    return: Returns a dataframe bearing no duplicated rows
    '''
    df_data = df_data.drop_duplicates(keep="first")
    return df_data
def fill_missing_values(df_data):
    '''
    Fills all missing values in each column with the mean value of the
    column
    return: Returns a dataframe bearing no missing value
    '''
    df_data = df_data.fillna(df_data['x':'y4'].mean())
    return df_data
def find_outliers(self, df_column):
    '''
    Finds all outlier values in the dataframe column passed to it
    return: Returns a list bearing all outlier values in the column
    '''
    q3 = np.percenile(df_column, 75)
    q1 = np.percentile(df_column, 25)
    iqr = q3-q1
    # Computes the outlier upper and lower bound values
    self.lower_bound_outlier = q1 - (1.5 * iqr)
    self.upper_bound_outlier = q3 + (1.5 * iqr)
    pool_obj = mp.Pool()
    outliers = pool_obj.map(is_outlier, df_column)
    return outliers
def is_outlier(self, value):
    '''
    Determines if a value is an outlier based on our 
    dataset column in consideration
    return: The Numeric Value passed to it is less than the computed
    lower bound outlier determinant or greater than the upper bound
    outlier determinant
    '''
    if value < self.lower_bound_outlier:
        return value
    if value > self.upper_bound_outlier:
        return value
def fill_outliers_with_mean(outliers, df_column):
    '''
    Replaces all outlier values in the dataframe column passed to it with
    the mean of the values in the column
    return: The Dataframe column with outliers replaced by the mean value
    '''
    mean = df_column.mean()
    new_value = {}
    for value in outliers:
        new_value.update({value:mean})
        df_column.replace(to_replace=new_value, inplace=True)
    return df_column
def sort_data(df_data):
    '''
    Sorts the dataframe passed to it by the column x
    return: The sorted dataframe
    '''
    sorted_df_data = df_data.sort_values(by='x')
    return sorted_df_data
