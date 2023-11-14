'''
Imports the needed modules for testing the class
'''
import unittest
import pandas as pd
import numpy as np
from data_exploration.data_wrangler import DataWrangler
from data_exploration.ols import OLS
class TestDataWrangler(unittest.TestCase):
    '''
    A simple DataWrangler Test class.
    Attributes:
        ols(OLS): an object of the OLS class, responsible for the fitting of 
        a linear regression model on the training dataset
        data_wrangler(DataWrangler): An object of the DataWrangler class, responsible
        for wrangling the dataset
        df_data(pandas.dataframe): Pandas dataframe object bearing the dataset
        to be wrangled.
        outlier_x(list): A list bearing outlier values found in column x of our 
        training dataset
        outlier_y1(list): A list bearing outlier values found in column y1 of our 
        training dataset
        outlier_y2(list): A list bearing outlier values found in column y2 of our 
        training dataset
        outlier_y3(list): A list bearing outlier values found in column y3 of our 
        training dataset
        outlier_y4(list): A list bearing outlier values found in column y4 of our 
        training dataset
    Methods:
        setUp(): Does the initialization of attributes needed to running 
        the test.
        test_load_data(): Tests the load_data() method of the class, that
        it truely creates an instance of a pandas dataframe.
        test_shape_of_data(): test the shape_data method that it successfully
        returns the shape of the dataframe constructed
        test_summary_statistics(): test the summary_statistics method that it successfully
        returns a dataframe bearing the summary statistics of each dataframe column
        test_find_missing_values(): test the find_missing_values method that it successfully
        returns a dataframe bearing the count of NaN values of each dataframe column
        test_duplicated_rows(): test the duplicated_rows method that it successfully
        returns a series bearing False for all elements
        test_drop_duplicated():test the drop_duplicated method that it successfully
        dropped all duplicated rows and that we have no change in our dataset
        shape since we have no duplicate rows
        test_fill_missing_values(self): test the fill_missing_vales method that it successfully
        filled missing values. In our dataset, we have no missing values,
        so we test that count of missing values is same before we 
        call the medthod and after we call the method.
        test_find_outliers():test the find outliers method that it works correctly in 
        finding outliers
        test_handle_outliers():test the method for filling outliers with the mean value
        test_sort_data():test the method for sorting the data        
    '''
    def setUp(self):
        '''
        Sets up the test via creating objects and variables needed
        for the running of the test
        '''
        self.ols = OLS("train.csv")
        self.ols.load_data()
        self.data_wrangler = DataWrangler(self.ols.dataframe)
        self.outlier_x = self.data_wrangler.find_outliers(self.data_wrangler.df_data.loc[:,'x'])
        self.outlier_y1 = self.data_wrangler.find_outliers(self.data_wrangler.df_data.loc[:,'y1'])
        self.outlier_y2 = self.data_wrangler.find_outliers(self.data_wrangler.df_data.loc[:,'y2'])
        self.outlier_y3 = self.data_wrangler.find_outliers(self.data_wrangler.df_data.loc[:,'y3'])
        self.outlier_y4 = self.data_wrangler.find_outliers(self.data_wrangler.df_data.loc[:,'y4'])
        
    def test_load_data(self):
        '''
        test the load_data method that it successfully
        constructed a dataframe from the .csv file
        '''     
        self.assertNotEqual(isinstance(self.data_wrangler.df_data, pd.DataFrame), False,
                            "The returned value is of type DataFrame")
    def test_shape_of_data(self):
        '''
        test the shape_data method that it successfully
        returns the shape of the dataframe constructed
        '''        
        df_shape= self.data_wrangler.shape_of_data()
        self.assertEqual(df_shape[0], 400, "The tuple contains at index 0, the value 400,"+
                        "which is our number of rows")
        self.assertEqual(df_shape[1], 5, "The tuple contains at index 1, the value 5,"+
                        "which is our number of columns")
    def test_summary_statistics(self):
        '''
        test the summary_statistics method that it successfully
        returns a dataframe bearing the summary statistics of each dataframe column
        '''
        df_summary = self.data_wrangler.summary_statistics()
        #Test Mean Values
        self.assertEqual(np.round(df_summary.loc['mean', 'x'],decimals=2), -0.05,
                         "The mean value of x column was truly computed")
        self.assertEqual(np.round(df_summary.loc['mean','y1'], decimals=2), -0.01,
                         "The mean value of y1 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['mean','y2'],decimals=2), 0.00,
                         "The mean value of y2 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['mean','y3'],decimals=2), -20.05,
                         "The mean value of y3 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['mean','y4'],decimals=2), 1.87,
                         "The mean value of y4 column was truly computed")  
        #Test Standard Deviations
        self.assertEqual(np.round(df_summary.loc['std', 'x'],decimals=2), 11.56,
                         "The std value of x column was truly computed")
        self.assertEqual(np.round(df_summary.loc['std', 'y1'],decimals=2), 0.74,
                         "The std value of y1 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['std', 'y2'],decimals=2), 0.73,
                         "The std value of y2 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['std', 'y3'],decimals=2), 3038.16,
                         "The std value of y3 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['std', 'y4'],decimals=2), 34.68,
                         "The std value of y4 column was truly computed")
        #Test Minimum Values
        self.assertEqual(np.round(df_summary.loc['min', 'x'],decimals=2), -20.00,
                         "The minimum value of x column was truly computed")
        self.assertEqual(np.round(df_summary.loc['min', 'y1'],decimals=2), -1.47,
                         "The minimum value of y1 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['min', 'y2'],decimals=2), -1.43,
                         "The minimum value of y2 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['min', 'y3'],decimals=2), -8020.18,
                         "The minimum value of y3 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['min', 'y4'],decimals=2), -57.80,
                         "The minimum value of y4 column was truly computed")
        #Test 25th Percentile Values
        self.assertEqual(np.round(df_summary.loc['25%', 'x'],decimals=2), -10.02,
                         "The 25th Percentile value of x column was truly computed")
        self.assertEqual(np.round(df_summary.loc['25%', 'y1'],decimals=2), -0.66,
                         "The 25th Percentile value of y1 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['25%', 'y2'],decimals=2), -0.59,
                         "The 25th Percentile value of y2 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['25%', 'y3'],decimals=2), -1017.92,
                         "The 25th Percentile value of y3 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['25%', 'y4'],decimals=2), -27.99,
                         "The 25th Percentile value of y4 column was truly computed")   
        #Test 50th Percentile Values
        self.assertEqual(np.round(df_summary.loc['50%', 'x'],decimals=2), -0.05,
                         "The 50th Percentile value of x column was truly computed")
        self.assertEqual(np.round(df_summary.loc['50%', 'y1'],decimals=2), 0.030,
                         "The 50th Percentile value of y1 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['50%', 'y2'],decimals=2), -0.020,
                         "The 50th Percentile value of y2 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['50%', 'y3'],decimals=2), -0.15,
                         "The 50th Percentile value of y3 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['50%', 'y4'],decimals=2), 1.79,
                         "The 50th Percentile value of y4 column was truly computed") 
        #Test 75th Percentile Values
        self.assertEqual(np.round(df_summary.loc['75%', 'x'],decimals=2), 9.93,
                         "The 75th Percentile value of x column was truly computed")
        self.assertEqual(np.round(df_summary.loc['75%', 'y1'],decimals=2), 0.64,
                         "The 75th Percentile value of y1 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['75%', 'y2'],decimals=2), 0.61,
                         "The 75th Percentile value of y2 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['75%', 'y3'],decimals=2), 987.71,
                         "The 75th Percentile value of y3 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['75%', 'y4'],decimals=2), 31.82,
                         "The 75th Percentile value of y4 column was truly computed")         
        #Test Maximum Values
        self.assertEqual(np.round(df_summary.loc['max', 'x'],decimals=2), 19.90,
                         "The Maximum value of x column was truly computed")
        self.assertEqual(np.round(df_summary.loc['max', 'y1'],decimals=2), 1.38,
                         "The Maximum value of y1 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['max', 'y2'],decimals=2), 1.42,
                         "The Maximum value of y2 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['max', 'y3'],decimals=2), 7900.73,
                         "The Maximum value of y3 column was truly computed")
        self.assertEqual(np.round(df_summary.loc['max', 'y4'],decimals=2), 61.91,
                         "The Maximum value of y4 column was truly computed")
    def test_find_missing_values(self):
        '''
        test the find_missing_values method that it successfully
        returns a dataframe bearing the count of NaN values of each dataframe column
        '''
        result = self.data_wrangler.find_missing_values()
        self.assertEqual(result.loc['x'], 0,"Column x has no missing values (Nan)")
        self.assertEqual(result.loc['y1'], 0,"Column y1 has no missing values (Nan)")
        self.assertEqual(result.loc['y2'], 0,"Column y2 has no missing values (Nan)")
        self.assertEqual(result.loc['y3'], 0,"Column y3 has no missing values (Nan)")
        self.assertEqual(result.loc['y4'], 0,"Column y4 has no missing values (Nan)")
    def test_duplicated_rows(self):
        '''
        test the duplicated_rows method that it successfully
        returns a series bearing False for all elements
        '''
        result = self.data_wrangler.duplicated_rows()
        self.assertEqual(result.all() == False,True, "No Row is a duplicated row") # pylint: disable=simplifiable-if-statement
    def test_drop_duplicated(self):
        '''
        test the drop_duplicated method that it successfully
        dropped all duplicated rows and that we have no change in our dataset
        shape since we have no duplicate rows
        '''
        result = self.data_wrangler.drop_duplicated()
        self.assertEqual(result.shape == (400,5), True,  "We still have the same shape of dataset")
    def test_fill_missing_values(self):
        '''
        test the fill_missing_vales method that it successfully
        filled missing values. In our dataset, we have no missing values,
        so we test that count of missing values is same before we 
        call the medthod and after we call the method.
        '''
        initial_result = self.data_wrangler.find_missing_values()
        self.data_wrangler.df_data = self.data_wrangler.fill_missing_values()
        later_result = self.data_wrangler.find_missing_values()
        self.assertEqual(initial_result.loc['x'] == later_result.loc['x'], True,
                         "Column x has no missing values (Nan)")
        self.assertEqual(initial_result.loc['y1'] == later_result.loc['y1'], True,
                         "Column y1 has no missing values (Nan)")
        self.assertEqual(initial_result.loc['y2'] == later_result.loc['y2'], True,
                         "Column y2 has no missing values (Nan)")
        self.assertEqual(initial_result.loc['y3'] == later_result.loc['y3'], True,
                         "Column y3 has no missing values (Nan)")
        self.assertEqual(initial_result.loc['y4'] == later_result.loc['y4'], True,
                         "Column y4 has no missing values (Nan)")
    def test_find_outliers(self):
        '''
        test the find outliers method that it works correctly in 
        finding outliers
        '''
        self.assertEqual(len(self.outlier_x) == 0, True,"No Outliers in Column X")
        self.assertEqual(len(self.outlier_y1) == 0, True,"No Outliers in Column Y1")
        self.assertEqual(len(self.outlier_y2) == 0, True,"No Outliers in Column Y2")
        self.assertEqual(len(self.outlier_y3) != 0, True,"No Outliers in Column Y3")        
        self.assertEqual(len(self.outlier_y4) == 0, True,"No Outliers in Column Y4")
    def test_handle_outliers(self):
        '''
        test the method for filling outliers with the mean value
        '''
        df_column_handled = self.data_wrangler.handle_outliers(self.outlier_y3, 
                                                               self.data_wrangler.df_data.loc[:,'y3'])
        new_outlier_y3 = self.data_wrangler.find_outliers(df_column_handled)
        self.assertEqual(len(new_outlier_y3) != 0, True,"Outlier values in column y3 "+
                         "are not handled, because ratio of data points that are outliers "+
                         "to total number of datapoints in the column are greater 2%,"+
                         "hence they aren't outliers") #The value 2% was chosen at my own discretion
    def test_sort_data(self):
        '''
        test the method for sorting the data
        '''
        sorted_df_data = self.data_wrangler.sort_data()
        self.assertEqual(sorted_df_data['x'].is_monotonic_increasing, True,
                         "Dataframe is truly sorted")     
if __name__ == '__main__':
    unittest.main()
