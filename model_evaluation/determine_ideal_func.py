'''
Imports the external modules needed for 
the class
'''
import numpy as np
from data_exploration.ols import OLS
class DetermineIdealFunctions(object):
    '''
    A simple DetermineIdealFunctions class.
    A class that  determines the four ideal functions out of the 50 
    ideal functions in the dataset ideal.csv.
    Attributes:
        df(pandas.dataframe): A pandas dataframe object bearing the dataset of 
        ideal functions, that we are to determine the four best for our created
        model.
        predicted_values(list): A 2-D list bearing the dataset generated from our 
        created model.
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
        get_ideal_dataframe():Returns the dataframe of the ideal functions dataset
        loaded
        get_existing_max_devia():Returns the exisiting maximum deviation of the
        created model from the training dataset
        sum_of_deviations(): computes the sum of squared residuals for the different
        models created for y1, y2, y3 and y4.
        calculated_max_deviation(): uses the numpy max function to determine the
        maximum sum of deviation - existing maximum deviation of the calculated
        regression.
        sum_of_devia_ideal_func(): determines the average sum of squared deviation of the
        predictions of the models created from each ideal function passed to it.
        determine _four_ideal(): determines the 4 ideal functions with the least sum of 
        deviation  
    Function:
        compute_pred_ideal_devia(y_pred, y_ideal): computes the deviation of ideal
        function dataset y_ideal from the prediction dataset y_pred
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
        ols_ideal = OLS(file_name1)
        ols_ideal.load_data()
        ols_training = OLS(file_name2)
        ols_training.load_data()
        ols_training.fit_regression()        
        self.df = ols_ideal.get_dataframe()
        self.residuals = ols_training.compute_residuals()
        self.predicted_values = ols_training.prepare_predicted_value()
        self.res1 = np.square(self.residuals[0])
        self.res2 = np.square(self.residuals[1])
        self.res3 = np.square(self.residuals[2])
        self.res4 = np.square(self.residuals[3])
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
        self.existing_max_devia = np.max(np.array(
            self.sum_of_deviation_val))
    def sum_of_deviation(self):
        '''
        Computes the sum of squared deviation - residuals for y1, y2
        y3 and y4 and updates the list sum_of_deviation_val
        '''
        for value in self.res1:
            self.sum_of_deviation_val[0] = self.sum_of_deviation_val[0] + np.square(value)
        for value in self.res2:
            self.sum_of_deviation_val[1] = self.sum_of_deviation_val[1] + np.square(value)
        for value in self.res3:
            self.sum_of_deviation_val[2] = self.sum_of_deviation_val[2] + np.square(value)
        for value in self.res4:
            self.sum_of_deviation_val[3] = self.sum_of_deviation_val[3] + np.square(value)
    def sum_of_devia_ideal_func(self):
        '''
        Computes the sum of deviation of each ideal function 
        from our prediction y1, y2, y3, and y4 using 
        the compute deviation function below
        '''
        for j in range(50):
            column = 'y'+str(j+1)
            ideal = self.df.loc[:,column]
            sum_of_squared_deviation = compute_pred_ideal_devia(
                np.array(self.predicted_values), np.array(ideal))
            self.sum_of__ideal_deviation.insert(j, sum_of_squared_deviation)
    def determine_four_ideal(self):
        '''
        Determines the best four ideal functions using the
        sum of deviation of each ideal function from the 
        predicted dataset from our model created using the 
        training dataset.
        returns: It returns a list of the ideal functions.
        '''
        ideal = ['','','','']
        for i in range(4):
            minimum = np.min(self.sum_of__ideal_deviation)
            index = self.sum_of__ideal_deviation.index(minimum)
            ideal[i] = "y"+str(index+1)
            self.sum_of__ideal_deviation.remove(minimum)
        return ideal
def compute_pred_ideal_devia(y_pred, y_ideal):
    '''
    Determines the sum of squared deviation of the ideal 
    function dataset y_ideal from the prediction dataset y_pred.
    y_ideal: The ideal function dataset to compute its deviation from the 
    prediction dataset
    y_pred: The prediction dataset to be used to compute the deviation
    of y_ideal dataset from it.
    returns: the sum of deviation
    '''
    temp_sum = 0
    total_sum = 0
    i =0
    for m in y_pred:
        for value in y_ideal:
            deviation = np.square(value-m[i])
            temp_sum = temp_sum + deviation
            i = i+1
        total_sum = total_sum + deviation
        temp_sum = 0
        i=0
    return total_sum
        
    
        
    
    
        
        
        
        
        
        
        