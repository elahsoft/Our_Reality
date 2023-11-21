'''
Imports the needed modules for testing the class
'''
import unittest
import copy
from model_evaluation.determine_ideal_func import DetermineIdealFunctions
class TestDetermineIdealFinctions(unittest.TestCase):
    '''
    A simple Test Class for the DetermineIdealFunctions  Class
    Attributes:
        deter_ideal_func(DetermineIdealFunctions): an object of the DetermineIdealFunctions class, 
        that grants us access to methods of the class.
        expected_sum_squared_devia(list): A list that bears the expected values of the 
        self.sum_of__ideal_deviation variable if sum_of_devia_ideal_func() did the 
        computation correctly.       
                
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
        self.deter_ideal_func = DetermineIdealFunctions("ideal.csv", "train2.csv")     
        self.deter_ideal_func.sum_of_deviation()   
        self.deter_ideal_func.calculated_max_deviation()
        self.deter_ideal_func.sum_of_devia_ideal_func()
        self.expected_sum_squared_devia = [3782.15559744031, 3833.065591154061, 
                                           2822.7303639295114, 3238.5346494921537, 
                                           5928.348227731232, 4030.0999556086645, 
                                           4029.735680184478, 3879.151718512194, 
                                           3764.992180678466, 2628.5454569444273, 
                                           2643.7713848483763, 10314.753698349263, 
                                           3774.9246018618496, 8330.541717926477, 
                                           5089.368104525439, 574615.4321372602, 
                                           687782.1617655145, 2399919.108523133, 
                                           605267.3953701552, 1028996.8800476314, 
                                           247293256.55195427, 247293256.55195427, 
                                           249545274.4715565, 991413334.777765, 
                                           2233309195.2556, 439793734.1351774, 
                                           132399843.21973655, 248546588.5675877, 
                                           272830336.9474602, 200265798.58495894, 
                                           2643.7713848483763, 3345.3207709445696, 
                                           2645.877477498237, 3748.7958808688118, 
                                           2880.6081519534273, 29287.190718692167, 
                                           8330.541717926477, 3850.73306482366, 
                                           577243.285314628, 28005.126560787357, 
                                           4629.401508848373, 15811.622185339631, 
                                           3511.5691300409694, 3904.900991087008, 
                                           2721.989939144335, 3258.865982329427, 
                                           4771.519743029628, 3872.4840358652523, 
                                           3782.15559744031, 3835.8775576715116]
    def test_sum_of_deviation(self):
        '''
        Tests the sum_of_deviation method.
        Sum of squared deviation should be close
        in range since the four functions are all 
        sinusoidal in shape - sine function
        '''
        self.assertEqual(self.deter_ideal_func.sum_of_deviation_val,
                         [1285.4770530972846, 1281.4513989704842, 
                          0.0, 3619.8862663683417],
                         "The sum of squared deviation of each model "+
                         "created from the training dataset is actually computed")
    def test_calculated_max_deviation(self):
        '''
        Tests the calculated_max_deviation method
        '''
        self.assertEqual(self.deter_ideal_func.existing_max_devia,
                         3619.8862663683417, "The maximum existing squared "+
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