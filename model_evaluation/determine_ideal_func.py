import numpy as np
import pandas as pd
from numba import jit
import statsmodels.api as sm
from data_exploration.ols import OLS
class DetermineIdealFunctions(OLS):
    '''
    A simple DetermineIdealFunctions class.
    A class that extends the OLS class and determines the four 
    ideal functions out of the 50 ideal functions in the dataset ideal.csv.
    Attributes:
        df(pandas.dataframe): A pandas dataframe object bearing the dataset of 
        ideal functions, that we are to determine the four best for our created
        model.
        predicted_values(list): A 2-D list bearing the dataset generated from our 
        created model.
        model(list): A list bearing the linear regression model created for y1,
        y2, y3, and y4
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
        load_dataset(filename): Loads the file ideal.csv into a dataframe
        sum_of_deviations(): computes the sum of squared residuals for the different
        models created for y1, y2, y3 and y4.
        calculated_max_deviation(): uses the numpy max function to determine the
        maximum sum of deviation - existing maximum deviation of the calculated
        regression.
        sum_of_devia_ideal_func(): determines the average sum of squared deviation of the
        predictions of the models created from each ideal function passed to it.
        compute_pred_ideal_devia(y_pred, y_ideal): computes the deviation of ideal
        function dataset y_ideal from the prediction dataset y_pred
        determine _four_ideal(): determines the 4 ideal functions with the least sum of 
        deviation  
    '''
    def __init__(self, file_name):
        '''
        The constructor method of the class.
        file_name: The file name of the file bearing the 50 ideal
        functions dataset
        '''
        super().__init__(file_name)
        super().load_data()
        self.df = super().get_dataframe()
        self.predicted_values = super().prepare_predicted_value()
        self.model = None
        self.res1 = None
        self.res2 = None
        self.res3 = None
        self.res4 = None
        self.existing_max_devia = None        
        self.sum_of_deviation_val = [0,0,0,0]
        self.sum_of__ideal_deviation = []
    def calculated_max_deviation(self):
        '''
        Determines the maximum out of the sum of 
        deviation of each predictive model created
        from the training dataset
        '''
        self.existing_max_devia = np.max(self.sum_of_deviation_val)
    @jit(nopython = True)
    def sum_of_deviation(self):
        '''
        Computes the sum of squared deviation - residuals for y1, y2
        y3 and y4 and updates the list sum_of_deviation_val
        '''
        self.model = super().fit_regression()
        residuals = super().compute_residuals()
        self.res1 = np.square(residuals[0])
        self.res2 = np.square(residuals[1])
        self.res3 = np.square(residuals[2])
        self.res4 = np.square(residuals[3])
        for value in self.res1:
            self.sum_of_deviation_val[0] = self.sum_of_deviation_val[0] + value
        for value in self.res2:
            self.sum_of_deviation_val[1] = self.sum_of_deviation_val[1] + value
        for value in self.res3:
            self.sum_of_deviation_val[2] = self.sum_of_deviation_val[2] + value
        for value in self.res4:
            self.sum_of_deviation_val[3] = self.sum_of_deviation_val[3] + value
    @jit(nopython = True)
    def sum_of_devia_ideal_func(self):
        '''
        Computes the sum of deviation of each ideal function 
        from our prediction y1, y2, y3, and y4 using 
        the compute deviation function below
        '''
        j = 0
        while j < 50:
            column = 'y'+str(j+1)
            ideal = self.df.loc[:,column]
            self.sum_of__ideal_deviation.insert(j,
                self.compute_pred_ideal_devia(self.predicted_values, ideal))
    @jit(nopython=True)
    def compute_pred_ideal_devia(self,y_pred, y_ideal):
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
            total_sum = total_sum + temp_sum
            temp_sum = 0
        return total_sum
    def determine_four_ideal(self):
        '''
        Determines the best four ideal functions using the
        sum of deviation of each ideal function from the 
        predicted dataset from our model created using the 
        training dataset.
        returns: It returns a list of the ideal functions.
        '''
        ideal = ['','','','']
        i = 0
        while i < 4:
            minimum = np.min(self.sum_of__ideal_deviation)
            index = self.sum_of__ideal_deviation.index(minimum)
            ideal[i] = "y"+str(index+1)
            self.sum_of__ideal_deviation.remove(minimum)
            return ideal
        
    
        
    
    
        
        
        
        
        
        
        