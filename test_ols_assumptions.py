'''
Imports the needed modules for testing the class
'''
import unittest
from data_exploration.ols_assumptions import OLSAssumptions
from data_exploration.data_wrangler import DataWrangler
from data_exploration.ols import OLS
class TestOLSAssumptions(unittest.TestCase):
    '''
    A simple Test Class for the OLSAssumptions  Class
    Attributes:
        ols(OLS): an object of the OLS class, responsible for the fitting of 
        a linear regression model on the training dataset
        data_wrangler(DataWrangler): An object of the DataWrangler class, responsible
        for wrangling the dataset
        df_data(pandas.dataframe): Pandas dataframe object bearing the dataset
        to be wrangled.
        ols_assump(OLSAssumptions): An object of the OLSAssumptions class that
        we use to access the methods of the class for testing.
        
    Methods:
        setUp(): Does the initialization of attributes needed to running 
        the test.
        test_check_heterocedasticity(): Tests the check_heterocedasticity method.
        test_check_normality(): Tests the check_normality method that creates a
        q-q plot of residuals for each Y value - y1, y2, y3, y4. It also uses a 
        Uses Jarque-Bera Test to check for the Normality of the dataset.
        test_check_linearity(): Tests the check_linearity method that creates a 
        scatter plot of Fitted Values Against Residuals for each Y value - y1, 
        y2, y3, y4. It also uses a rainbow test to further check for linearity assumption
        test_check_independence(): Tests the check_independence method that creates
        a scatter plot of residuals against order of observation for each Y value - 
        y1, y2, y3, y4; and also Uses Durbin-Watson Statistics to test for independence.
        The scatter plot should show a random scatter of points with no clear upward 
        or downward trend.        
    '''
    def setUp(self):
        '''
        Sets up the test via creating objects and variables needed
        for the running of the test
        '''
        self.ols = OLS("train.csv")
        self.ols.load_data()
        self.data_wrangler = DataWrangler(self.ols.dataframe)
        self.df_data = self.data_wrangler.df_data
        self.ols_assump = OLSAssumptions("train.csv")
    def test_check_heterocedasticity(self):
        '''
        Tests the check_heterocedasticity method.
        A list with all False values indicates
        homocedasticity, hence no transformation
        of y values needed
        '''
        result = self.ols_assump.check_heterocedasticity()
        print("Heterocedascity P-value (<0.05):", self.ols_assump.p_value)
        self.assertEqual(result[0],False,"variance of the residuals of y1 is "
                         +"constant across the range of predictor values")
        self.assertEqual(result[1],False,"variance of the residuals of y2 is "
                         +"constant across the range of predictor values")
        self.assertEqual(result[2],False,"variance of the residuals of y3 is "
                         +"constant across the range of predictor values")
        self.assertEqual(result[3],False,"variance of the residuals of y4 is "
                         +"constant across the range of predictor values")
    def test_check_normality(self):
        '''
        Tests the check_normality method that creates a
        q-q plot of residuals for each Y value - y1, y2, y3, y4. 
        It also uses a Uses Jarque-Bera Test to check 
        for the Normality of the dataset
        '''
        result = self.ols_assump.check_normality()
        print("Normality P-value (>0.05):", self.ols_assump.jb_p_value)
        self.assertEqual(result[0], True,
                         "Y1 residuals are normally distributed, inspect visually "+
                         "to be sure.")
        self.assertEqual(result[1], True,
                         "Y2 residuals are normally distributed, inspect visually "+
                         "to be sure.")
        self.assertEqual(result[2], True,
                         "Y3 residuals are normally distributed, inspect visually "+
                         "to be sure.")
        self.assertEqual(result[3], True,
                         "Y4 residuals are normally distributed, inspect visually "+
                         "to be sure.")
    def test_check_linearity(self):
        '''
        Tests the check_linearity method that creates a 
        scatter plot of Fitted Values Against Residuals
        for each Y value - y1, y2, y3, y4. It also uses a 
        rainbow test to further check for linearity assumption
        '''
        result = self.ols_assump.check_linearity()
        print("Linearity P-value (< 0.05):", self.ols_assump.rainbow_p_value)
        self.assertEqual(result[0], True,
                         "Y1 Scatter Plot of Fitted Values Against Residuals"+
                         " might not show a pattern, inspect visually "+
                         "to be sure.")
        self.assertEqual(result[1], True,
                         "Y2 Scatter Plot of Fitted Values Against Residuals"+
                         " might not show a pattern, inspect visually "+
                         "to be sure.")
        self.assertEqual(result[2], True,
                         "Y3 Scatter Plot of Fitted Values Against Residuals"+
                         " might not show a pattern, inspect visually "+
                         "to be sure.")
        self.assertEqual(result[3], True,
                         "Y4 Scatter Plot of Fitted Values Against Residuals"+
                         " might not show a pattern, inspect visually "+
                         "to be sure.")
    def test_check_independence(self):
        '''
        Tests the check_independence method that creates
        a scatter plot of residuals against order of observation
        for each Y value - y1, y2, y3, y4; and also 
        Uses Durbin-Watson Statistics to test for independence.
        The scatter plot should show a random scatter of points
        with no clear upward or downward trend.
        '''
        result = self.ols_assump.check_independence()
        print("Independence dw statistics (dw_statistics1 >= 1.5 and dw_statistics1 <= 2.5):",
              self.ols_assump.dw_statistics)
        self.assertEqual(result[0], True,
                         "Y1 Scatter Plot of residuals against order of observation"+
                         " might not show a pattern, inspect visually "+
                         "to be sure.")
        self.assertEqual(result[1], True,
                         "Y2 Scatter Plot of residuals against order of observation"+
                         " might not show a pattern, inspect visually "+
                         "to be sure.")
        self.assertEqual(result[2], True,
                         "Y3 Scatter Plot of residuals against order of observation"+
                         " might not show a pattern, inspect visually "+
                         "to be sure.")
        self.assertEqual(result[3], True,
                         "Y4 Scatter Plot of residuals against order of observation"+
                         " might not show a pattern, inspect visually "+
                         "to be sure.")
    if __name__ == '__main__':
        unittest.main()
        