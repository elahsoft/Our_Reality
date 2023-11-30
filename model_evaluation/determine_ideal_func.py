'''
Imports the external modules needed for 
the class
'''
import numpy as np
import pandas as pd
from data_exploration.ols import OLS
import config.config as cf
from utility.dataframe_utility import DataFrameUtility
class DetermineIdealFunctions(DataFrameUtility):
    '''
    A simple DetermineIdealFunctions class.
    A class that  determines the four ideal functions out of the 50 
    ideal functions in the dataset ideal.csv. A child class of DataFrameUtility
    Attributes:
        df(pandas.dataframe): A pandas dataframe object bearing the dataset of 
        ideal functions, that we are to determine the four best for our created
        model.
        train(list): A 2-D list bearing the training dataset for y1, y2, y3 and y4.
        train_df(pandas.dataframe): A pandas dataframe object bearing the training
        dataset.
        residuals(list): A list bearing the residuals of the linear regression 
        model created for y1, y2, y3, and y4
        res1(list): A list bearing the residuals of of the model we created with respect
        to the training dataset independent variable x and dependent variable y1
        res2(list): A list bearing the residuals of of the model we created with respect
        to the training dataset independent variable x and dependent variable y2
        res3(list): A list bearing the residuals of of the model we created with respect
        to the training dataset independent variable x and dependent variable y3
        res4(list): A list bearing the residuals of of the model we created with respect
        to the training dataset independent variable x and dependent variable y4
        existing_max_devia(list): A numeric list that bears the maximum squared
        deviation existing between the models created with respect to the training dataset 
        for the n rows of dataset.        
        existing_deviation_val(list): A list that bears the squared deviation of each 
        dependent variable in the training datasetfrom the predictive model gotten from 
        the fit.
        sum_of__ideal_deviation(list): A list bearing the sum of squared deviation of 
        each ideal function from the four predictive model
        max_ideal_train_devia(list): A list that bears indices of the selected ideal 
        functions with the maximum mean squared deviation from the training dataset
    Methods:
        get_existing_max_devia():Returns the exisiting maximum deviation of the
        created model from the training dataset
        squared_deviation(): Computes the squared deviation of the each fitted functions 
        from the train dataset and calls the calculated_max_deviation method to pick the 
        maximum out of the four deviation values.
        calculated_max_deviation(integer): uses the numpy max function to determine the
        maximum squared of deviation - existing maximum deviation of the calculated
        regression out of the four values in the list existing_deviation_val and insert it 
        to a list existing_max_devia.
        sum_of_devia_ideal_func(): determines the average sum of squared deviation of the
        training dataset from each ideal function passed to it.
        determine _four_ideal(): determines the 4 ideal functions with the least sum of 
        deviation
        determine_max_ideal_train_devia(list): Determines the selected ideal functions with the 
        maximum mean squared deviation from the training dataset for each row of the dataset and 
        updates max_ideal_train_devia instance variable with the index of the selected ideal 
        functions that have the mean squared deviation as the maximum. 
    Function:
        compute_train_ideal_devia(y_train, y_ideal): computes the deviation of ideal
        function dataset y_ideal from the training dataset y_train
    '''
    def __init__(self, file_name1, file_name2):
        '''
        The constructor method of the class.
        file_name1: The file name of the file bearing the 50 ideal
        functions dataset
        file_name2: The file name of training dataset to be used to 
        create the model for which four best ideal functions is to
        be determined
        '''
        super().__init__(file_name1)
        super().load_data()
        ols = OLS(file_name2)
        ols.load_data()
        ols.fit_regression([True, True, True, True])      
        self.df = super().get_dataframe()
        self.residuals = ols.compute_residuals([True, True, True, True])
        self.train = [ols.dataframe.loc[:,'y1'],
                      ols.dataframe.loc[:,'y2'],
                      ols.dataframe.loc[:,'y3'],
                      ols.dataframe.loc[:,'y4']]
        self.train_df = ols.get_dataframe()
        self.res1 = np.square(self.residuals[0])
        self.res2 = np.square(self.residuals[1])
        self.res3 = np.square(self.residuals[2])
        self.res4 = np.square(self.residuals[3])
        self.existing_max_devia = []
        self.existing_deviation_val = [0,0,0,0]
        self.sum_of__ideal_deviation = []
        self.max_ideal_train_devia = []
    def get_ideal_dataframe(self):
        '''
        Returns the dataframe of the ideal functions dataset
        loaded
        '''
        return self.df
    def get_existing_max_devia(self):
        '''
        Returns the exisiting maximum deviation of the
        created model from the training dataset
        '''
        return self.existing_max_devia
    def calculated_max_deviation(self, row_number):
        '''
        Determines the maximum out of the squared
        deviation of each predictive model created
        from the training dataset and insert it to a list
        row_number: The row number that we are computing
        the existing maximum deviation.
        '''
        self.existing_max_devia.insert(row_number, np.max(np.array(
            self.existing_deviation_val)))
    def squared_deviation(self):
        '''
        Computes the squared deviation of the each fitted functions 
        from the train dataset and calls the calculated_max_deviation
        method to pick the maximum out of the four deviation values.
        '''
        df_res = pd.read_csv(cf.INPUT_FILE_PATH+"residuals.csv")
        for count in range(len(df_res.loc[:,'res1'])):
            self.existing_deviation_val.insert(0, 
                    np.square(df_res.loc[:,'res1'][count]))
            self.existing_deviation_val.insert(1, 
                    np.square(df_res.loc[:,'res2'][count]))
            self.existing_deviation_val.insert(2, 
                    np.square(df_res.loc[:,'res3'][count]))
            self.existing_deviation_val.insert(3, 
                    np.square(df_res.loc[:,'res4'][count]))
            self.calculated_max_deviation(count)
        existing_max_devia_df = pd.DataFrame(data={
            'max_devia': self.existing_max_devia
        })
        super().write_to_file(existing_max_devia_df, "existing_max_devia.csv")
    def sum_of_devia_ideal_func(self):
        '''
        Computes the sum of deviation of each ideal function 
        from our training dataset y1, y2, y3, and y4 using 
        the compute deviation function below
        '''
        for j in range(50):
            column = 'y'+str(j+1)
            ideal = self.df.loc[:,column]
            sum_of_squared_deviation = compute_train_ideal_devia(
                np.array(self.train), np.array(ideal))
            self.sum_of__ideal_deviation.insert(j, sum_of_squared_deviation)
    def determine_four_ideal(self):
        '''
        Determines the best four ideal functions using the
        sum of deviation of each ideal function from the 
        training dataset.
        returns: It returns a list of the ideal functions.
        '''
        #sorts the list and picks the four minimum values
        indices = sorted(range(len(self.sum_of__ideal_deviation)), key=lambda i:
            self.sum_of__ideal_deviation[i])[:4]
        ideal = ["y"+str(indices[0]+1),"y"+str(indices[1]+1),"y"+str(indices[2]+1),
                "y"+str(indices[3]+1)] 
        sel_ideal_df = pd.DataFrame(data={
            ideal[0]: self.df.loc[:,ideal[0]],
            ideal[1]: self.df.loc[:,ideal[1]],
            ideal[2]: self.df.loc[:,ideal[2]],
            ideal[3]: self.df.loc[:,ideal[3]]
        })
        super().write_to_file(sel_ideal_df, "selected_ideal.csv")
        return ideal
    def determine_max_ideal_train_devia(self, ideal):
        '''
        Determines the selected ideal functions with the 
        maximum mean squared deviation from the training dataset for
        each row of the dataset and updates max_ideal_train_devia 
        instance variable with the index of the selected ideal functions
        that have the mean squared deviation as the maximum.
        ideal: A list of the selected ideal functions dataset 
        '''
        ideal1 = self.df.loc[:,ideal[0]]
        ideal2 = self.df.loc[:,ideal[1]]
        ideal3 = self.df.loc[:,ideal[2]]
        ideal4 = self.df.loc[:,ideal[3]]
        for count in range(400):
            #compute the mean deviation of eacj ideal function from the 
            #training dataset for the row number - count in consideration. 
            ideal1_devia = (np.square(ideal1[count]-self.train[0][count])+
                                 np.square(ideal1[count]-self.train[1][count])+ 
                                 np.square(ideal1[count]-self.train[2][count])+
                                 np.square(ideal1[count]-self.train[3][count]))/4
            ideal2_devia = (np.square(ideal2[count]-self.train[0][count])+
                                 np.square(ideal2[count]-self.train[1][count])+
                                 np.square(ideal2[count]-self.train[2][count])+
                                 np.square(ideal2[count]-self.train[3][count]))/4
            ideal3_devia = (np.square(ideal3[count]-self.train[0][count])+
                                 np.square(ideal3[count]-self.train[1][count])+
                                 np.square(ideal3[count]-self.train[2][count])+
                                 np.square(ideal3[count]-self.train[3][count]))/4
            ideal4_devia = (np.square(ideal4[count]-self.train[0][count])+
                                 np.square(ideal4[count]-self.train[1][count])+
                                 np.square(ideal4[count]-self.train[2][count])+
                                 np.square(ideal4[count]-self.train[3][count]))/4
            row_devia = [ideal1_devia, ideal2_devia, ideal3_devia, ideal4_devia]
            max_value = np.max(row_devia)
            index = row_devia.index(max_value)
            self.max_ideal_train_devia.insert(count, index)        
def compute_train_ideal_devia(y_train, y_ideal):
    '''
    Determines the squared deviation of the ideal 
    function dataset y_ideal from the train dataset y_train.
    y_ideal: The ideal function dataset to compute its deviation from the 
    training dataset
    y_train: The training dataset to be used to compute the deviation
    of y_ideal dataset from it.
    returns: the sum of deviation
    '''
    temp_sum = 0
    total_sum = 0
    i =0
    for m in y_train:
        for value in y_ideal:
            deviation = np.square(value-m[i])
            temp_sum = temp_sum + deviation
            i = i+1
        total_sum = total_sum + temp_sum
        temp_sum = 0
        i=0
        #To normalize, since the sum was the sum of squared deviation between ideal function 
        # in consideration and the four training functions y1, y2, y3, and y4
        total_sum = total_sum/4
    return total_sum  