'''
Imports the modules needed by the class
'''
import numpy as np
import pandas as pd
from model_evaluation.determine_ideal_func import DetermineIdealFunctions
from data_exploration.ols import OLS
class EvaluateModel(DetermineIdealFunctions):
    """
    A simple EvaluateModel class that mapps the four selected ideal functions
    to the test data and also compute the test data deviation from the four 
    ideal functions.
    
    Attributes:
        ideal_dataset(pandas.dataframe): It houses the dataframe bearing the ideal 
        functions dataset.
        existing_max_devia(float): It houses the value of the existing maximum 
        deviation of the linear regression models created from the training dataset
        four_ideal_func(list): It houses a list of the four selected ideal functions 
        for our model created with OLS method.
        ols_test(OLS): An object of OLS class, that is used for accessing the methods
        for reading the test data and loading it into a dataframe.
        test_df(pandas.dataframe): A pandas dataframe object containing the test dataset
        mapping(list): A 2-D list that houses the deviation of test dataset from the selected 
        ideal functions that passed the mapping criteria that difference between the 
        maximum of the calculated deviation of the models created from the training dataset
        (column wise) and the row wise maximum deviation of the test dataset from the four
        selected ideal functions not being more than a sqrt(2).
        max(list): A 2-D list that bears the index of the four selected functions that 
        have the maximum deviation from the test data (row wise)
        ideal_test_devia(list): A 2-D list that bears the squared deviation value of the 
        selected four ideal functions from the test dataset.
        mapped(pandas.dataframe): A dataframe bearing the output of the mapping of the
        selected four ideal functions to the test dataset, with columns: x, y, ideal_1_deviation,
        ideal_2_deviation, ideal_3_deviation, ideal_4_deviation, Number_of_Ideal_Function 
        
    Methods:
        extract_selected_ideal_data(): Extracts the corresponding selected four ideal function 
        equivalents of the test dataset. returns: a 2-D bearing the corresponding selected 
        four ideal function equivalents of the test dataset.
        compute_test_devia_from_ideal(list): Computes the squared deviation of the test dataset
        from its corresponding dataset for the four selected ideal functions.
        determine_max_deviation(): Determines the maximum deviation out of the four
        deviation values of each row of test data from the four selected ideal functions. 
        Creates a 2-D list bearing the column index of the deviations of the 
        four selected ideal functions that is maximum out of the four deviation values.
        map_ideal(): A function that creates a 2-D list that houses the deviation of test dataset 
        from the selected ideal functions that passed the mapping criteria that difference 
        between the maximum of the calculated deviation of the models created from the training 
        dataset (column wise), that is normalized by dividing by the number of rows in the training
        dataset, and the row wise maximum deviation of the test dataset from the four
        selected ideal functions not being more than a sqrt(2).
        construct_mapping_df(): A function that creates a dataframe bearing the output of the mapping 
        of the selected four ideal functions to the test dataset, with columns: x, y, ideal_1_deviation, 
        ideal_2_deviation, ideal_3_deviation, ideal_4_deviation, Number_of_Ideal_Function.        
        
    Usage:
        evaluateModel = new EvaluateModel("train.csv", "ideal.csv", "test.csv)
        evaluateModel.extract_selected_ideal_data()
        evaluateModel.compute_test_devia_from_ideal(self.selected_ideal_dataset)
        evaluateModel.determine_max_deviation()
        evaluateModel.map_ideal()
        evaluateModel.construct_mapping_df()
        print("output of Mapping", evaluateModel.mapped)
    """
    def __init__(self, file_name1, file_name2, file_name3):
        super().__init__(file_name1, file_name2)
        super().sum_of_deviation()
        super().calculated_max_deviation()
        self.ideal_dataset = super().get_ideal_dataframe()
        #Divided by the number of rows of column x to normalize
        #the value for comparison with maximum deviation of selected
        #ideal functions from the test data.
        self.existing_max_devia = super().get_existing_max_devia()/len(
            self.get_ideal_dataframe().loc[:,'x'])
        self.four_ideal_func = super().determine_four_ideal()
        self.ols_test = OLS(file_name3)
        self.ols_test.load_data()
        self.test_df = self.ols_test.dataframe
        self.mapping = []
        self.max = []
        self.ideal_test_devia = [[],[],[],[]]
        self.mapped = None
    def extract_selected_ideal_data(self):
        '''
        Extracts the corresponding selected four ideal function 
        equivalents of the test dataset
        returns: a 2-D bearing the corresponding selected four ideal function 
        equivalents of the test dataset
        '''
        col = self.test_df.loc[:,'x']
        #the corresponding selected four ideal functions dataset
        #for the test data
        selected_ideal_dataset = [[],[],[],[]]
        i=0
        for value in col:
            selected_ideal_dataset[i].insert(0,
                self.ideal_dataset.loc[self.ideal_dataset['x'] == value,
                                                      self.four_ideal_func[0]].value)
            selected_ideal_dataset[i].insert(1,
                self.ideal_dataset.loc[self.ideal_dataset['x'] == value,
                                                      self.four_ideal_func[1]].value)
            selected_ideal_dataset[i].insert(2,
                self.ideal_dataset.loc[self.ideal_dataset['x'] == value,
                                                      self.four_ideal_func[2]].value)
            selected_ideal_dataset[i].insert(3,
                self.ideal_dataset.loc[self.ideal_dataset['x'] == value,
                                                      self.four_ideal_func[3]].value)
            i = i+1
        return selected_ideal_dataset
    def compute_test_devia_from_ideal(self, ideal_equivalence):
        '''
        Computes the squared deviation of the test dataset
        from its corresponding dataset for the four selected
        ideal functions
        '''
        col_x = self.test_df.loc[:,'y']
        i = 0
        for value in col_x:
            self.ideal_test_devia[i][0].insert(
            np.square(value-ideal_equivalence[i][0]))
            self.ideal_test_devia[i][1].insert(
            np.square(value-ideal_equivalence[i][1]))
            self.ideal_test_devia[i][2].insert(
            np.square(value-ideal_equivalence[i][2]))
            self.ideal_test_devia[i][3].insert(
            np.square(value-ideal_equivalence[i][3]))
            i = i+1
    def determine_max_deviation(self):
        '''
        Determines the maximum deviation out of the four
        deviation values of each row of test data from the 
        four selected ideal functions. Creates a 2-D list 
        bearing the column index of the deviations of the 
        four selected ideal functions that is maximum out 
        of the four deviation values.
        '''
        deviation = self.ideal_test_devia
        col_x = self.test_df.loc[:,'x']
        i = 0
        for i in range(len(col_x)):
            maximum = np.max(deviation[i][0], deviation[i][1],
                         deviation[i][2], deviation[i][3])
            #extract the index of the of all deviation of 
            #four selected ideal functions from the test 
            #data for the specific value of x in consideration
            # that is equal to the maximum above 
            #Index is a list of indices
            current_devia_row = [deviation[i][0], deviation[i][1],
                                 deviation[i][2], deviation[i][3]]
            index = [j for j, k in enumerate(current_devia_row) if k == maximum]
            self.max.insert(i, index)
            i = i+1
    def map_ideal(self):
        '''
        A function that creates a 2-D list that houses the deviation of test dataset 
        from the selected ideal functions that passed the mapping criteria that difference 
        between the maximum of the calculated deviation of the models created from the training 
        dataset (column wise), that is normalized by dividing by the number of rows in the training
        dataset, and the row wise maximum deviation of the test dataset from the four
        selected ideal functions not being more than a sqrt(2).
        '''
        i = 0
        j = 0
        col_x = self.test_df.loc[:,'x']
        for i in range(len(col_x)):
            dev = [0,0,0,0]
            while j < len(self.max[i]):
                diff = self.calculated_max_deviation - self.ideal_test_devia[
                    j][self.max[i][j]]
                if np.abs(diff) < np.sqrt(2):
                    dev.insert(self.max[i][j],self.ideal_test_devia[i][
                        self.max[i][j]])
                    self.mapping.insert(i, dev)
    def construct_mapping_df(self):
        '''
        A function that creates a dataframe bearing the output of the mapping 
        of the selected four ideal functions to the test dataset, with columns: 
        x, y, ideal_1_deviation, ideal_2_deviation, ideal_3_deviation, ideal_4_deviation, 
        Number_of_Ideal_Function
        '''
        dictionary = {}
        dictionary.update({'x': self.test_df.loc[:,'x']})
        dictionary.update({'y': self.test_df.loc[:,'y']})
        i = 0
        ideal_1_devia = []
        ideal_2_devia = []
        ideal_3_devia = []
        ideal_4_devia = []
        no_of_ideal = []
        for value in self.mapping:
            ideal_1_devia.insert(i, value[0])
            ideal_2_devia.insert(i, value[1])
            ideal_3_devia.insert(i, value[2])
            ideal_4_devia.insert(i, value[3])
            no_of_ideal.insert(i, len(self.max[i]))
            i = i+1
        dictionary.update({self.four_ideal_func[0]:ideal_1_devia})
        dictionary.update({self.four_ideal_func[1]:ideal_2_devia})
        dictionary.update({self.four_ideal_func[2]:ideal_3_devia})
        dictionary.update({self.four_ideal_func[3]:ideal_4_devia})
        dictionary.update({'Number_of_Ideal_Function': no_of_ideal})
        self.mapped = pd.DataFrame(data=dict)    