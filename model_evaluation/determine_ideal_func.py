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
        existing_max_devia(float): A numeric value that is the maximum sum of squared
        deviation existing between the models created with respect to the training dataset        
        sum_of_deviation_val(list): A list that bears the sum of squared deviation of each 
        dependent variable from the predictive model gotten from the fit.
        sum_of__ideal_deviation(list): A list bearing the sum of squared deviation of 
        each ideal function from the four predictive model
    Methods:
        get_existing_max_devia():Returns the exisiting maximum deviation of the
        created model from the training dataset
        sum_of_deviations(): computes the sum of squared residuals for the different
        models created for y1, y2, y3 and y4.
        calculated_max_deviation(): uses the numpy max function to determine the
        maximum sum of deviation - existing maximum deviation of the calculated
        regression.
        sum_of_devia_ideal_func(): determines the average sum of squared deviation of the
        training dataset from each ideal function passed to it.
        determine _four_ideal(): determines the 4 ideal functions with the least sum of 
        deviation  
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
        self.res_df = pd.DataFrame(data={
            "res1": self.residuals[0],
            "res2": self.residuals[1],
            "res3": self.residuals[2],
            "res4": self.residuals[3]
        })
        super().write_to_file(self.res_df, "residuals.csv")
        self.existing_max_devia = []
        self.sum_of_deviation_val = [0,0,0,0]
        self.sum_of__ideal_deviation = []
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
    def calculated_max_deviation(self):
        '''
        Determines the maximum out of the sum of 
        deviation of each predictive model created
        from the training dataset
        '''
        #Divided by the number of rows of training dataset to normalize
        #the value for comparison with maximum deviation of selected
        #ideal functions from the test data.
        self.existing_max_devia = np.max(np.array(
            self.sum_of_deviation_val))/400
    def sum_of_deviation(self):
        '''
        Computes the sum of squared deviation of the each fitted functions 
        from the train dataset and takes the mean to normalize it, 
        and updates the list sum_of_deviation_val
        '''
        df_res = pd.read_csv(cf.INPUT_FILE_PATH+"residuals.csv")
        sum_value=0
        for value in df_res.loc[:,'res1']:
            sum_value = sum_value + np.square(value)
        self.sum_of_deviation_val[0] = sum_value
        sum_value=0
        for value in df_res.loc[:,'res2']:
            sum_value = sum_value + np.square(value)
        self.sum_of_deviation_val[1] = sum_value
        sum_value=0
        for value in df_res.loc[:,'res3']:
            sum_value = sum_value + np.square(value)
        self.sum_of_deviation_val[2] = sum_value
        sum_value=0
        for value in df_res.loc[:,'res4']:
            sum_value = sum_value + np.square(value)
        self.sum_of_deviation_val[3] = sum_value
        
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
def compute_train_ideal_devia(y_train, y_ideal):
    '''
    Determines the sum of squared deviation of the ideal 
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