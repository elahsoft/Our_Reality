'''
Imports the needed modules for testing the class
'''
import unittest
from utility.dataframe_utility import DataFrameUtility
class TestOLSAssumptions(unittest.TestCase):
    ''' A simple Test Class for the OLSAssumptions  Class
    Attributes:
        ols(OLS): 
        
    Methods:
        setUp(): Does the initialization of attributes needed for running 
        the test.
        test_check_heterocedasticity(): 
    '''
    def setUp(self):
        '''
        Sets up the test via creating objects and variables needed
        for the running of the test
        '''
        self.dataframe_utility = DataFrameUtility("train.csv")
        self.dataframe_utility.load_data()
    def test_load_data(self):
        '''
        Tests the load_data method to ensure that
        it actually read the file and created a dataframe 
        from it.
        '''
        df = self.dataframe_utility.dataframe
        self.assertEqual(df.head() is not None, True,"The dataframe wasn't created ".join(
            "from the file."))
    def test_get_dataframe(self):
        '''
        Tests the get_dataframe method to ensure that
        it actually retrieves the dataframe
        '''
        df = self.dataframe_utility.get_dataframe()
        self.assertEqual(df.head() is not None, True,"The dataframe wasn't retrieved successfully!")
if __name__ == '__main__':
    unittest.main()
    