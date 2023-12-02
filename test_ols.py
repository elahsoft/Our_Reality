'''
Imports the needed modules for testing the class
'''
import unittest
from data_exploration.ols import OLS
class TestOLSAssumptions(unittest.TestCase):
    '''
    A simple Test Class for the OLS  Class
    Attributes:
        ols(OLS): an object of the OLS class, responsible for the fitting of 
        a linear regression model on the training dataset
                
    Methods:
        setUp(): Does the initialization of attributes needed for running.
        test_fit_regression(): Tests that the fit_regression method called 
        in setUp method actually created the models.
        test_compute_residuals(): Tests the compute_residuals method that it 
        created the list of residuals of the same number of rows as the training
        dataset.
        test_prepare_predicted_value(): Tests the prepare_predicted_value method
        that the list of predicted values are correctly created from the created 
        models and that they have the same number of rows as the training dataset.
        
    '''
    def setUp(self):
        '''
        Sets up the test via creating objects and variables needed
        for the running of the test
        '''
        self.ols = OLS("train.csv")
        self.ols.load_data()
        self.ols.fit_regression([True, True, True, True])
    def test_fit_regression(self):
        '''
        tests the fit regression method
        '''
        self.assertEqual(len(self.ols.model) ==0, False, "The fit regression method did "+
                         "not create the model.")
    def test_compute_residuals(self):
        '''
        tests the compute_residuals method
        '''
        self.ols.compute_residuals([True, True, True, True])
        self.assertEqual(len(self.ols.residuals) == 0, False, "The compute residuals method "+
                         "did not compute the residuals")
        self.assertEqual(len(self.ols.residuals[0]), len(self.ols.dataframe.loc[:,'x']),
                         "The compute residuals method did not compute the residuals")
        self.assertEqual(len(self.ols.residuals[1]), len(self.ols.dataframe.loc[:,'x']),
                         "The compute residuals method did not compute the residuals")
        self.assertEqual(len(self.ols.residuals[2]), len(self.ols.dataframe.loc[:,'x']),
                         "The compute residuals method did not compute the residuals")
        self.assertEqual(len(self.ols.residuals[3]), len(self.ols.dataframe.loc[:,'x']),
                         "The compute residuals method did not compute the residuals")
    def test_prepare_predicted_value(self):
        '''
        tests the prepare_predicted_value method
        '''
        pred_values = self.ols.prepare_predicted_value([True, True, True, True])
        self.assertEqual(len(pred_values) == 0, False, "The prepare_predicted_value method "+
                         "did not compute the predicted values from the fitted model")
        self.assertEqual(len(pred_values[0]), len(self.ols.dataframe.loc[:,'x']),
                         "The returned predicted values by the prepare_predicted_value method "+
                         "did not match the number of data points fed to it.")
        self.assertEqual(len(pred_values[1]), len(self.ols.dataframe.loc[:,'x']),
                          "The returned predicted values by the prepare_predicted_value method "+
                         "did not match the number of data points fed to it.")
        self.assertEqual(len(pred_values[2]), len(self.ols.dataframe.loc[:,'x']),
                          "The returned predicted values by the prepare_predicted_value method "+
                         "did not match the number of data points fed to it.")
        self.assertEqual(len(pred_values[3]), len(self.ols.dataframe.loc[:,'x']),
                          "The returned predicted values by the prepare_predicted_value method "+
                         "did not match the number of data points fed to it.")
    if __name__ == '__main__':
        unittest.main()
        