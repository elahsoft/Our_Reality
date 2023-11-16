import unittest
from model_evaluation.evaluate_model import EvaluateModel
class TestEvaluateModel(unittest.TestCase):
    '''
    A simple Test Class for the EvaluateModel  Class
    Attributes:
              
                
    Methods:
        setUp(): Does the initialization of attributes needed for running 
        the test.
               
    '''
    def setUp(self):
        '''
        Sets up the test via creating objects and variables needed
        for the running of the test
        '''
        self.evaluate_model = EvaluateModel("training.csv", "ideal.csv")     
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