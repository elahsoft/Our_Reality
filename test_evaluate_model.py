import unittest
from model_evaluation.evaluate_model import EvaluateModel
class TestEvaluateModel(unittest.TestCase):
    '''
    A simple Test Class for the EvaluateModel  Class
    Attributes:
              
                
    Methods:
        setUp(): Does the initialization of attributes needed for running 
        the test.
        test_compute_test_devia_from_ideal(): Test the compute_test_devia_from_ideal method
        test_determine_max_deviation(): Test the determine_max_deviation method
        test_map_ideal(): Test the map_ideal method
        test_construct_mapping_df(): Test the construct_mapping_df method      
    '''
    def setUp(self):
        '''
        Sets up the test via creating objects and variables needed
        for the running of the test
        '''
        self.evaluate_model = EvaluateModel("train.csv", "ideal.csv", "test.csv") 
        self.evaluate_model.compute_train_devia_from_ideal()
        self.evaluate_model.determine_max_deviation()
        self.evaluate_model.map_ideal()
        self.evaluate_model.construct_mapping_df()
    def test_compute_test_devia_from_ideal(self):
        '''
        Test the compute_test_devia_from_ideal method
        '''
        self.assertEqual(len(self.evaluate_model.ideal_train_devia) == 4, True,
                         "The Deviation of the selected ideal functions from the train data was "+
                         "not computed well!")
        self.assertEqual(len(self.evaluate_model.ideal_train_devia[0]) == 400, True,
                         "The Deviation of the selected ideal functions from the train data was "+
                         "not computed well!")
        self.assertEqual(len(self.evaluate_model.ideal_train_devia[1]) == 400, True,
                         "The Deviation of the selected ideal functions from the train data was "+
                         "not computed well!")
        self.assertEqual(len(self.evaluate_model.ideal_train_devia[2]) == 400, True,
                         "The Deviation of the selected ideal functions from the train data was "+
                         "not computed well!")
        self.assertEqual(len(self.evaluate_model.ideal_train_devia[3]) == 400, True,
                         "The Deviation of the selected ideal functions from the train data was "+
                         "not computed well!")
    def test_determine_max_deviation(self):
        '''
        Test the determine_max_deviation method
        '''
        self.assertEqual(len(self.evaluate_model.max) == 400, True, 
                         "The maximum deviation out of the "+
                         " the ideal function deviations from the train data was "+
                         "computed well!")
    def test_map_ideal(self):
        '''
        Test the map_ideal method
        '''
        self.assertEqual(len(self.evaluate_model.mapping) == 400, True,
                         "The map_ideal function did not do the mapping") 
    def test_construct_mapping_df(self):
        '''
        Test the construct_mapping_df method
        '''
        self.assertEqual(self.evaluate_model.mapped.head(5) is not None, True,
                         "The dataframe of the final resulf of the mapping "+
                         "was not properly created!")
        index = self.evaluate_model.mapped.columns
        self.assertEqual(len(self.evaluate_model.mapped.loc[:,index[0]]) == 100,
                         True, "The dataframe does not bear the exact number of test rows")
        self.assertEqual(len(self.evaluate_model.mapped.loc[:,index[1]]) == 100,
                         True, "The dataframe does not bear the exact number of test rows")
        self.assertEqual(len(self.evaluate_model.mapped.loc[:,index[2]]) == 100,
                         True, "The dataframe does not bear the exact number of test rows")
        self.assertEqual(len(self.evaluate_model.mapped.loc[:,index[3]]) == 100,
                         True, "The dataframe does not bear the exact number of test rows")
        self.assertEqual(len(self.evaluate_model.mapped.loc[:,index[4]]) == 100,
                         True, "The dataframe does not bear the exact number of test rows")
        self.assertEqual(len(self.evaluate_model.mapped.loc[:,index[5]]) == 100,
                         True, "The dataframe does not bear the exact number of test rows")
        self.assertEqual(len(self.evaluate_model.mapped.loc[:,index[6]]) == 100,
                         True, "The dataframe does not bear the exact number of test rows")
if __name__ == '__main__':
    unittest.main() 