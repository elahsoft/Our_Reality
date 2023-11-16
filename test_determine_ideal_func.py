'''
Imports the needed modules for testing the class
'''
import unittest
import copy
from model_evaluation.determine_ideal_func import DetermineIdealFunctions
class TestDetermineIdealFinctions(unittest.TestCase):
    '''
    A simple Test Class for the DetermineIdealFinctions  Class
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
        setUp(): Does the initialization of attributes needed for running 
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
        self.deter_ideal_func = DetermineIdealFunctions("ideal.csv", "train.csv")     
        self.deter_ideal_func.sum_of_deviation()   
        self.deter_ideal_func.calculated_max_deviation()
        self.deter_ideal_func.sum_of_devia_ideal_func()
        self.expected_sum_squared_devia = [22919981.35999242, 22923572.558442272, 
                                           22823456.835509147, 22875195.478353556, 
                                           23034275.95094429, 22936812.603422407, 
                                           22936789.00249287, 22926759.56475184, 
                                           22918753.12395203, 22762115.579261508, 
                                           22736960.07590698, 22345168.939361874, 
                                           22595699.229817934, 23122995.945057575, 
                                           22995997.966678172, 19714632.752833806, 
                                           27396746.54893075, 17755462.89598534, 
                                           19649719.615861293, 18941958.314106748, 
                                           194906724.09452176, 194906724.09452176, 
                                           347780788.63685083, 863715779.0089693, 
                                           2030252833.625338, 362341813.80526936, 
                                           210134158.69836563, 195969881.56074643, 
                                           216659330.9567933, 155400430.6438924, 
                                           22736960.07590698, 22885205.137777787, 
                                           22735765.38202372, 22917585.691204783, 
                                           22832762.012879495, 22002905.29760578, 
                                           23122995.945057575, 22924801.298539575, 
                                           19708968.804526754, 22020613.73903049, 
                                           22544625.3514451, 23318275.893438216, 
                                           22899421.5793236, 22928515.062167894, 
                                           22803066.898016, 22877164.19763894, 
                                           22979847.895948198, 22926302.092695817, 
                                           22919981.35999242, 22923768.71628605]
        self.ideal_func = None
    def test_sum_of_deviation(self):
        '''
        Tests the sum_of_deviation method.
        Sum of squared deviation should be close
        in range since the four functions are all 
        sinusoidal in shape - sine function
        '''
        self.assertEqual(self.deter_ideal_func.sum_of_deviation_val,
                         [216.4744967687914, 218.35154938213591, 
                          2020940685348972.8, 4.801266630283453],
                         "The sum of squared deviation of each model "+
                         "created from the training dataset is actually computed")
    def test_calculated_max_deviation(self):
        '''
        Tests the calculated_max_deviation method
        '''
        self.assertEqual(self.deter_ideal_func.existing_max_devia,
                         2020940685348972.8, "The maximum existing squared "+
                         "deviation between the training dataset and "+
                         "the model created is correctly determined")
    def test_sum_of_devia_ideal_func(self):
        '''
        Tests the sum_of_devia_ideal_func
        '''
        for j in range(50):
            ideal = "y"+str(j+1)
            self.assertEqual(self.deter_ideal_func.sum_of__ideal_deviation[j], 
                             self.expected_sum_squared_devia[j], "Ideal "+ideal+" = "+
                             str(self.deter_ideal_func.sum_of__ideal_deviation[j])
                             +" sum of squared deviation computed does not equal "+
                             str(self.expected_sum_squared_devia[j]))
    def test_determine_four_ideal(self):
        '''
        Test the determine_four_ideal method
        '''
        #Prevents changes from altering original list
        sum_of_ideal_devia = copy.deepcopy(
            self.deter_ideal_func.sum_of__ideal_deviation)
        ideal = self.deter_ideal_func.determine_four_ideal()
        index_min_1 = int(ideal[0][1:]) - 1
        index_min_2 = int(ideal[1][1:]) - 1
        index_min_3 = int(ideal[2][1:]) - 1
        index_min_4 = int(ideal[3][1:]) - 1
        self.assertEqual(sum_of_ideal_devia[index_min_1],
                         self.expected_sum_squared_devia[index_min_1],
                         "Ideal Function y"+str(index_min_1+1)+
                         " Determined as Minimum is Incorrect")
        self.assertEqual(sum_of_ideal_devia[index_min_2],
                         self.expected_sum_squared_devia[index_min_2],
                         "Ideal Function y"+str(index_min_2+1)+
                         " Determined as Minimum is Incorrect")
        self.assertEqual(sum_of_ideal_devia[index_min_3],
                         self.expected_sum_squared_devia[index_min_3],
                         "Ideal Function y"+str(index_min_3+1)+
                         " Determined as Minimum is Incorrect")
        self.assertEqual(sum_of_ideal_devia[index_min_4],
                         self.expected_sum_squared_devia[index_min_4],
                         "Ideal Function y"+str(index_min_4+1)+
                         " Determined as Minimum is Incorrect")
if __name__ == '__main__':
    unittest.main()   