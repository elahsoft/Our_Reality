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
                
    Methods:
        setUp(): Does the initialization of attributes needed for running 
        the test.
        test_sum_of_deviation(): Tests the sum_of_deviation method.
        test_calculated_max_deviation(): Tests the calculated_max_deviation method
        test_sum_of_devia_ideal_func():Tests the sum_of_devia_ideal_func

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
    def test_sum_of_deviation(self):
        '''
        Tests the sum_of_deviation method.
        '''
        self.assertEqual(len(self.deter_ideal_func.sum_of_deviation_val),
                         4,
                         "The sum of squared deviation of each model created  "+
                         "from the training dataset was not actually computed")
    def test_calculated_max_deviation(self):
        '''
        Tests the calculated_max_deviation method
        '''
        self.assertEqual(self.deter_ideal_func.existing_max_devia,
                         10803.824460227288, "The maximum existing squared "+
                         "deviation between the training dataset and "+
                         "the model created was not correctly determined")
    def test_sum_of_devia_ideal_func(self):
        '''
        Tests the sum_of_devia_ideal_func
        '''
        for j in range(50):
            ideal = "y"+str(j+1)
            self.assertEqual(self.deter_ideal_func.sum_of__ideal_deviation[j] > 0, 
                             True, "Ideal "+ideal+" = "+
                             str(self.deter_ideal_func.sum_of__ideal_deviation[j])
                             +" sum of squared deviation computed does not equal was not computed "+
                             "correctly!")
    def test_determine_four_ideal(self):
        '''
        Test the determine_four_ideal method
        '''
        #Prevents changes from altering original list
        sum_of_ideal_devia = copy.deepcopy(
            self.deter_ideal_func.sum_of__ideal_deviation)
        ideal = self.deter_ideal_func.determine_four_ideal()
        self.assertEqual(len(ideal) == 4, True, "Lenght of selected "+
                         "ideal functions list is not up to 4")
        #computes the index of the selected ideal function
        index_min_1 = int(ideal[0][1:]) - 1
        index_min_2 = int(ideal[1][1:]) - 1
        index_min_3 = int(ideal[2][1:]) - 1
        index_min_4 = int(ideal[3][1:]) - 1
        self.assertEqual(sum_of_ideal_devia[index_min_1] is not None,
                         True, "Ideal Function y"+str(index_min_1+1)+
                         " Sum of Deviation was correctly retrieved")
        self.assertEqual(sum_of_ideal_devia[index_min_2] is not None,
                         True, "Ideal Function y"+str(index_min_2+1)+
                         " Sum of Deviation was correctly retrieved")
        self.assertEqual(sum_of_ideal_devia[index_min_3] is not None,
                         True, "Ideal Function y"+str(index_min_3+1)+
                         " Sum of Deviation was correctly retrieved")
        self.assertEqual(sum_of_ideal_devia[index_min_4] is not None,
                         True, "Ideal Function y"+str(index_min_4+1)+
                         " Sum of Deviation was correctly retrieved")
if __name__ == '__main__':
    unittest.main()   