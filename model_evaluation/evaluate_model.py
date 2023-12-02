'''
Imports the modules needed by the class
'''
import numpy as np
import pandas as pd
from model_evaluation.determine_ideal_func import DetermineIdealFunctions
from data_exploration.ols import OLS
from utility.dataframe_utility import DataFrameUtility
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
        ideal_train_devia(list): A 2-D list that bears the squared deviation value of the 
        selected four ideal functions from the test dataset.
        mapped(pandas.dataframe): A dataframe bearing the output of the mapping of the
        selected four ideal functions to the test dataset, with columns: x, y, ideal_1_deviation,
        ideal_2_deviation, ideal_3_deviation, ideal_4_deviation, Number_of_Ideal_Function 
        
    Methods:
        compute_train_devia_from_ideal(list): Computes the squared deviation of the train dataset
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
        selected_ideal_dataset = evaluateModel.extract_selected_ideal_data()
        evaluateModel.compute_test_devia_from_ideal(selected_ideal_dataset)
        evaluateModel.determine_max_deviation()
        evaluateModel.map_ideal()
        evaluateModel.construct_mapping_df()
        print("output of Mapping", evaluateModel.mapped)
    """
    def __init__(self, file_name1, file_name2, file_name3):
        super().__init__(file_name2, file_name1)
        super().squared_deviation()
        super().sum_of_devia_ideal_func()
        self.ideal_dataset = super().get_ideal_dataframe()        
        self.existing_max_devia = super().get_existing_max_devia()
        self.four_ideal_func = super().determine_four_ideal()
        super().determine_max_ideal_train_devia(self.four_ideal_func)
        self.ols_test = OLS(file_name3)
        self.test_df = self.ols_test.dataframe
        self.mapping = []
        self.mapped = None
        self.ideal_train_devia_unflattened = []
    def compute_train_devia_from_ideal(self):
        '''
        Computes the squared deviation of the test dataset
        from its corresponding dataset for the four selected
        ideal functions
        '''
        selec_ideal_1_list = []
        selec_ideal_2_list = []
        selec_ideal_3_list = []
        selec_ideal_4_list = []
        i = 0
        while i < 400:
            selec_ideal_1_list.insert(i,self.ideal_train_devia[i][0])
            selec_ideal_2_list.insert(i,self.ideal_train_devia[i][1])
            selec_ideal_3_list.insert(i,self.ideal_train_devia[i][2])
            selec_ideal_4_list.insert(i,self.ideal_train_devia[i][3])
            i = i+1
        self.ideal_train_devia_unflattened = [selec_ideal_1_list, selec_ideal_2_list,
                                 selec_ideal_3_list, selec_ideal_4_list]
        #write to a file
        ideal_train_devia_df = pd.DataFrame(data={
            self.four_ideal_func[0]+"_res":
                selec_ideal_1_list,
            self.four_ideal_func[1]+"_res":
                selec_ideal_2_list,
            self.four_ideal_func[2]+"_res":
                selec_ideal_3_list,
            self.four_ideal_func[3]+"_res":
                selec_ideal_4_list
        })         
        super().write_to_file(ideal_train_devia_df, "ideal_train_residuals.csv")        
    def map_ideal(self):
        '''
        A function that creates a 2-D list that houses the deviation of train dataset 
        from the selected ideal functions that passed the mapping criteria that the maximum 
        of the calculated deviation of the models created from the training 
        dataset (row wise) does not exceed the row wise maximum deviation of the four
        selected ideal functions from the training dataset by more than factor sqrt(2).
        N/B: The task pdf said the existing maximum deviation of the computed regression
        should be greater than the largest deviation between ideal function and training 
        dataset by factor of sqrt(2) for the ideal function to be mapped.
        Obviously, since we added polynomial features to capture complexities and also 
        employed regularization to prevent overfitting, our deviation values are very small 
        as compared to the deviation of the ideal functions from the training dataset as we 
        observed, which is obviously what would happen, because two functions are sinusoidal, one 
        appearing like a log and the other a straight line graph, so any ideal function that can 
        be mapped to a single function representing the four of them must be chosen by considering 
        the four of them on the average scale, and for that case, the deviation of the selected ideal 
        for such a function representing the four of them would be as connoted by the four dependent 
        variables of the training dataset would have a very large deviation value, so based on this in 
        the routine below for mapping, we mapped using the condition that the existing maximum deviation 
        for each row of the training dataset must be less than the largest deviation of the selected 
        ideal functions from the training dataset by factor of sqrt(2).
        '''
        j = 0
        for ind in self.max_ideal_train_devia:
            dev = [None,None,None,None]
            for i in ind:
                if self.existing_max_devia[j] * np.sqrt(2) < self.ideal_train_devia[j][i]:
                    dev[i] = self.ideal_train_devia[j][i]
            j = j+1
            self.mapping.insert(j, dev)
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
        for value in self.test_df.loc[:,'x']:
            index = self.train_df.index[self.train_df['x'] == value].tolist()
            ideal_1_devia.insert(i, self.mapping[index[0]][0])
            ideal_2_devia.insert(i, self.mapping[index[0]][1])
            ideal_3_devia.insert(i, self.mapping[index[0]][2])
            ideal_4_devia.insert(i, self.mapping[index[0]][3])
            zero_count = self.mapping[index[0]].count(None)
            number_of_ideal = len(self.mapping[index[0]]) - zero_count
            no_of_ideal.insert(i, number_of_ideal)
        dictionary.update({"Delta_"+self.four_ideal_func[0]:ideal_1_devia})
        dictionary.update({"Delta_"+self.four_ideal_func[1]:ideal_2_devia})
        dictionary.update({"Delta_"+self.four_ideal_func[2]:ideal_3_devia})
        dictionary.update({"Delta_"+self.four_ideal_func[3]:ideal_4_devia})
        dictionary.update({'Number_of_Ideal_Function': no_of_ideal})
        self.mapped = pd.DataFrame(data=dictionary)
        dataframe_utility = DataFrameUtility()
        dataframe_utility.write_to_file(self.mapped, "mapping.csv")