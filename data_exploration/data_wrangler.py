'''
Imports all the modules needed by the class.
'''
import numpy as np
class DataWrangler(object):
    """
    A simple DataWrangler class.
    Attributes:
        df_data(pandas.dataframe): The pandas dataframe object bearing the dataset 
        to be wrangled
        self.lower_bound_outlier(Integer): An integer indicating the lower bound
        determinant of outliers
        self.upper_bound_outlier(Integer): An integer indicating the upper bound 
        determinant of outliers
    Methods:
        shape_of_data(df_data): Returns the shape of the dataframe passed to it.
        summary_statistics(column_data): Returns a summary statistics of pandas dataframe
        passed to it.
        find_missing_values(df_data): Returns a list of missing values (nan) by columns.
        duplicated_rows(df_data): Returns duplicated rows
        drop_duplicates(df_data): Returns a pandas dataframe with duplicated rows dropped.
        fill_missing_values(df_data): Fills missing values (nan) in the dataframe with the mean 
        value of each column
        find_outliers(df_column): Returns all values that are outliers in the numeric 
        column passed to it.
        is_outlier(value): Returns the value if it is an outlier
        handle_outliers(outliers, df_column): fills all entries in the dataframe 
        column passed to it, that are outliers with the mean value of the column.
        sort_data(df_data): Sorts the dataframe by the 'x' column    

        Example:
            data_wrangler = DataWrangler(file_name="train.csv")
            df_data = data_wrangler.load_data()
            print(df_data.head(5))  # Output: The first five rows of the train dataset
    """
    def __init__(self, df_data):
        '''
        Constructor for the DataWrangler Class
        df_data: The dataframe for wrangling
        '''
        self.df_data = df_data
        self.lower_bound_outlier = 0
        self.upper_bound_outlier = 0
   
    def shape_of_data(self):
        '''
        Computes the shape of the dataframe passed to it
        return: A tuple, which is the shape of the dataframe.
        '''
        df_shape = self.df_data.shape
        return df_shape
    def summary_statistics(self):
        '''
        Computes the summary statistic of each column of the dataframe
        return: A dataframe, which gives a summary statistic of each
        column in the dataframe.
        '''
        return self.df_data.describe()
    def find_missing_values(self):
        '''
        Computes the count of missing values in each column of the dataframe
        return: Returns a pandas series bearing the details on the count of missing
        values in each column of the dataframe
        '''
        df_result = self.df_data.apply(lambda x: sum(x.isnull()), axis=0)
        return df_result
    def duplicated_rows(self):
        '''
        Finds all duplicated rows in the dataframe
        return: Returns a dataframe bearing duplicated rows in the dataframe
        '''
        df_duplicated = self.df_data.duplicated()
        return df_duplicated
    def drop_duplicated(self):
        '''
        Drops all duplicated rows in the dataframe and keeps the first
        occurrence
        return: Returns a dataframe bearing no duplicated rows
        '''
        df_data = self.df_data.drop_duplicates(keep="first")
        return df_data
    def fill_missing_values(self):
        '''
        Fills all missing values in each column with the mean value of the
        column
        return: Returns a dataframe bearing no missing value
        '''
        df_data = self.df_data.apply(lambda col: col.fillna(col.mean()))
        return df_data
    def find_outliers(self, df_column):
        '''
        Finds all outlier values in the dataframe column passed to it
        df_column: The column extracted from the dataframe that outliers
        existing in it are to be found.
        return: Returns a list bearing all outlier values in the column
        '''
        third_quar = np.percentile(df_column.to_numpy(), 75)
        first_quar = np.percentile(df_column, 25)
        iqr = third_quar - first_quar
        # Computes the outlier upper and lower bound values
        self.lower_bound_outlier = first_quar - (1.5 * iqr)
        self.upper_bound_outlier = third_quar + (1.5 * iqr)
        outliers = []
        i=0
        for value in df_column:
            status = self.is_outlier(value)
            if status == value:
                outliers.insert(i, value)
            i = i+1
        return outliers
    def is_outlier(self, value):
        '''
        Determines if a value is an outlier based on our 
        dataset column in consideration
        value: Value to be determined if it is an outlier
        return: The Numeric Value passed to it is less than the computed
        lower bound outlier determinant or greater than the upper bound
        outlier determinant
        '''
        if value < self.lower_bound_outlier:
            return value
        if value > self.upper_bound_outlier:
            return value
    def handle_outliers(self, outliers, df_column):
        '''
        Replaces all outlier values in the dataframe column passed to it with
        the mean of the values in the column. Checks if the number of datapoints that
        are outliers is like 2% of the number of rows, it handles it, else it returns
        the column without handling it.
        outliers: A list of values considered to be outliers in the column
        df_column extracted from the dataframe
        df_column: The column extracted from the dataframe, that we want to 
        replace outliers in it with the mean.
        return: The Dataframe column with outliers replaced by the mean value
        '''
        percentage_of_outliers = len(outliers)/len(df_column) * 100
        if percentage_of_outliers < 2 or percentage_of_outliers == 2:
            mean = df_column.mean()
            new_value = {}
            for value in outliers:
                new_value.update({value:mean})
                df_column.replace(to_replace=new_value, inplace=True)
                return df_column
        else:
            return df_column
    def sort_data(self):
        '''
        Sorts the dataframe passed to it by the column x
        return: The sorted dataframe
        '''
        sorted_df_data = self.df_data.sort_values(by='x')
        return sorted_df_data

