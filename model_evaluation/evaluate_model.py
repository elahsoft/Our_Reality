'''
Imports the modules needed by the class
'''
import numpy as np
import pandas as pd
from model_evaluation.determine_ideal_func import DetermineIdealFunctions
from data_exploration.ols import OLS
from data_exploration.data_wrangler import DataWrangler
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
        ideal_train_devia(list): A 2-D list that bears the squared deviation value of the 
        selected four ideal functions from the test dataset.
        mapped(pandas.dataframe): A dataframe bearing the output of the mapping of the
        selected four ideal functions to the test dataset, with columns: x, y, ideal_1_deviation,
        ideal_2_deviation, ideal_3_deviation, ideal_4_deviation, Number_of_Ideal_Function 
        
    Methods:
        extract_selected_ideal_data(): Extracts the corresponding selected four ideal function 
        equivalents of the train dataset. returns: a 2-D bearing the corresponding selected 
        four ideal function equivalents of the train dataset.
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
        super().sum_of_deviation()
        super().sum_of_devia_ideal_func()
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
        self.ideal_train_devia = []
        self.mapped = None
    def extract_selected_ideal_data(self):
        '''
        Extracts the corresponding selected four ideal function 
        equivalents of the training dataset
        returns: a 2-D bearing the corresponding selected four ideal function 
        equivalents of the training dataset
        '''
        col = self.train_df.loc[:,'x']
        #the corresponding selected four ideal functions dataset
        #for the test data
        selected_ideal_dataset = []
        i=0
        for value in col:
            temp_list = []
            temp_list.insert(0,
                self.ideal_dataset.loc[self.ideal_dataset['x'] == value,
                                                      self.four_ideal_func[0]].values[0])
            temp_list.insert(1,
                self.ideal_dataset.loc[self.ideal_dataset['x'] == value,
                                                      self.four_ideal_func[1]].values[0])
            temp_list.insert(2,
                self.ideal_dataset.loc[self.ideal_dataset['x'] == value,
                                                      self.four_ideal_func[2]].values[0])
            temp_list.insert(3,
                self.ideal_dataset.loc[self.ideal_dataset['x'] == value,
                                                      self.four_ideal_func[3]].values[0])
            selected_ideal_dataset.insert(i, temp_list)
            i = i+1
        return selected_ideal_dataset
    def compute_train_devia_from_ideal(self):
        '''
        Computes the squared deviation of the test dataset
        from its corresponding dataset for the four selected
        ideal functions
        '''
        #defined in the super class of EvaluateModel class
        col_y1 = self.train_df.loc[:,'y1']
        col_y2 = self.train_df.loc[:,'y2']
        col_y3 = self.train_df.loc[:,'y3']
        col_y4 = self.train_df.loc[:,'y4']
        selec_ideal_1_list = []
        selec_ideal_2_list = []
        selec_ideal_3_list = []
        selec_ideal_4_list = []
        i = 0
        for _, value in enumerate(self.ideal_dataset.loc[:,self.four_ideal_func[0]]):
            #computes the squared deviation of each selected ideal dataset
            #from the four train functions and then 
            #normalizes the value by dividing by four so that we 
            #compare to existing maximum deviation of fitted models
            #and be able to have the criteria of their difference
            # being within sqrt(2) possible.
            selec_ideal_1_list.insert(i,
            ((col_y1[i]-value)**2 + (col_y2[i]-value)**2 + (
                col_y3[i]-value)**2 + (col_y4[i]-value)**2)/4)
            i = i+1
        #print("selec_ideal_1_list.insert0", selec_ideal_1_list[0])
        #each row in it, is mean of sum of squared deviation for 
        # y1, y2, y3, y4 training dataset from the four 
        # ideal function. We find mean to normalize it. And we 
        #find deviation of each function from the ideal functions 
        #because no guideline was given as to which of the four 
        # ideal function should be used for y1, y2, y3, or y4
        #in computing the deviation
        self.ideal_train_devia.insert(0, selec_ideal_1_list)
        i = 0
        for _, value in enumerate(self.ideal_dataset.loc[:,self.four_ideal_func[1]]):
            #computes the squared deviation of each selected ideal dataset
            #from the four train functions and then 
            #normalizes the value by dividing by four so that we 
            #compare to existing maximum deviation of fitted models
            #and be able to have the criteria of their difference
            # being within sqrt(2) possible.
            selec_ideal_2_list.insert(i,
            ((col_y1[i]-value)**2 + (col_y2[i]-value)**2 + (
                col_y3[i]-value)**2 + (col_y4[i]-value)**2 )/4)
            i = i+1
        #print("selec_ideal_1_list.insert0", selec_ideal_1_list[0])
        #each row in it, is mean of sum of squared deviation for 
        # y1, y2, y3, y4 training dataset from the four 
        # ideal function. We find mean to normalize it. And we 
        #find deviation of each function from the ideal functions 
        #because no guideline was given as to which of the four 
        # ideal function should be used for y1, y2, y3, or y4
        #in computing the deviation
        self.ideal_train_devia.insert(1, selec_ideal_2_list)
        i = 0
        for _, value in enumerate(self.ideal_dataset.loc[:,self.four_ideal_func[2]]):
            #computes the squared deviation of each selected ideal dataset
            #from the four train functions and then 
            #normalizes the value by dividing by four so that we 
            #compare to existing maximum deviation of fitted models
            #and be able to have the criteria of their difference
            # being within sqrt(2) possible.
            devia = ((col_y1[i]-value)**2 + (col_y2[i]-value)**2 + (
                col_y3[i]-value)**2 + (col_y4[i]-value)**2)/4
            selec_ideal_3_list.insert(i, devia)
            i = i+1
        #print("selec_ideal_1_list.insert0", selec_ideal_1_list[0])
        #each row in it, is mean of sum of squared deviation for 
        # y1, y2, y3, y4 training dataset from the four 
        # ideal function. We find mean to normalize it. And we 
        #find deviation of each function from the ideal functions 
        #because no guideline was given as to which of the four 
        # ideal function should be used for y1, y2, y3, or y4
        #in computing the deviation
        self.ideal_train_devia.insert(2, selec_ideal_3_list)
        i = 0
        for _, value in enumerate(self.ideal_dataset.loc[:,self.four_ideal_func[3]]):
            #computes the squared deviation of each selected ideal dataset
            #from the four train functions and then 
            #normalizes the value by dividing by four so that we 
            #compare to existing maximum deviation of fitted models
            #and be able to have the criteria of their difference
            # being within sqrt(2) possible.
            selec_ideal_4_list.insert(i,
            ((col_y1[i]-value)**2 + (col_y2[i]-value)**2 + (
                col_y3[i]-value)**2 + (col_y4[i]-value)**2 )/4)
            i = i+1
        #print("selec_ideal_1_list.insert0", selec_ideal_1_list[0])
        #each row in it, is mean of sum of squared deviation for 
        # y1, y2, y3, y4 training dataset from the four 
        # ideal function. We find mean to normalize it. And we 
        #find deviation of each function from the ideal functions 
        #because no guideline was given as to which of the four 
        # ideal function should be used for y1, y2, y3, or y4
        #in computing the deviation        
        self.ideal_train_devia.insert(3, selec_ideal_4_list)
        #write to a file
        ideal_train_devia_df = pd.DataFrame(data={
            self.four_ideal_func[0]+"_res":
                self.ideal_train_devia[0],
            self.four_ideal_func[1]+"_res":
                self.ideal_train_devia[1],
            self.four_ideal_func[2]+"_res":
                self.ideal_train_devia[2],
            self.four_ideal_func[3]+"_res":
                self.ideal_train_devia[3]
        })
        data_wrangler = DataWrangler(ideal_train_devia_df)
        data_wrangler.write_to_file(ideal_train_devia_df,"ideal_train_residuals.csv")
        
    def determine_max_deviation(self):
        '''
        Determines the maximum deviation out of the four
        deviation values of each row of test data from the 
        four selected ideal functions. Creates a 2-D list 
        bearing the column index of the deviations of the 
        four selected ideal functions that is maximum out 
        of the four deviation values.
        '''
        deviation = self.ideal_train_devia
        col_x = self.train_df.loc[:,'x']
        i = 0
        for i in range(len(col_x)):
            maximum = np.max([deviation[0][i], deviation[1][i],
                         deviation[2][i], deviation[3][i]])
            #extract the index of the of all deviation of 
            #four selected ideal functions from the test 
            #data for the specific value of x in consideration
            # that is equal to the maximum above 
            #Index is a list of indices
            current_devia_row = [deviation[0][i], deviation[1][i],
                                 deviation[2][i], deviation[3][i]]
            index = [j for j, k in enumerate(current_devia_row) if k == maximum]
            self.max.insert(i, index)
    def map_ideal(self):
        '''
        A function that creates a 2-D list that houses the deviation of test dataset 
        from the selected ideal functions that passed the mapping criteria that difference 
        between the maximum of the calculated deviation of the models created from the training 
        dataset (column wise), that is normalized by dividing by the number of rows in the training
        dataset, and the row wise maximum deviation of the test dataset from the four
        selected ideal functions not being more than a sqrt(2).
        '''
        j = 0
        col_x = self.test_df.loc[:,'x']
        for ind in self.max:
            '''
            The criteria of existing maximum deviation of the calculated 
            regression does not exceed the largest deviation between training 
            dataset (A) and the ideal function (C) chosen for it by more than 
            factor sqrt(2) is dependent on how efficient our OLS class does the fitting,
            if the fitting is done efficient, then existing maximum deviation would be 
            less than largest deviation between training dataset (A) and the ideal 
            function (C). So we test for which is larger and know which to subtract
            from the other, so that we get a difference that would at least be less than 
            sqrt(2)
            '''
            dev = [None,None,None,None]
            for i in ind:
                diff = 0
                if self.existing_max_devia < self.ideal_train_devia[i][j]:
                    diff = self.existing_max_devia - self.ideal_train_devia[i][j]
                else:
                    diff = self.ideal_train_devia[i][j] - self.existing_max_devia
                if diff < np.sqrt(2):
                    dev[i] = self.ideal_train_devia[i][j]
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
            no_of_ideal.insert(i, len(self.max[index[0]]))
        dictionary.update({"Delta_"+self.four_ideal_func[0]:ideal_1_devia})
        dictionary.update({"Delta_"+self.four_ideal_func[1]:ideal_2_devia})
        dictionary.update({"Delta_"+self.four_ideal_func[2]:ideal_3_devia})
        dictionary.update({"Delta_"+self.four_ideal_func[3]:ideal_4_devia})
        dictionary.update({'Number_of_Ideal_Function': no_of_ideal})
        self.mapped = pd.DataFrame(data=dictionary)
        data_wrangler = DataWrangler(self.mapped)
        data_wrangler.write_to_file(self.mapped, "mapping.csv")