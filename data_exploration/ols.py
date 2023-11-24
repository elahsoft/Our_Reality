'''
Imports the modules needed by the class
'''
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from utility.dataframe_utility import DataFrameUtility
class OLS(DataFrameUtility):
    """
    A simple Linear Regression Fitting Class that uses Ordinary Least 
    Square Method for the fitting. It is a child class of DataFrameUtility
    
    Attributes:
        model(list): A list of models created for y1, y2, y3 and y4
        residuals(list): A  list bearing the residuals of y1, y2, y3, y4 with 
        regards to the training dataset used in creating the model
        
    Methods:
        fit_regression(): Fits a linear regression model over the data.
        compute_residuals(): Computes the residuals of the dataset
        prepare_predicted_value(): Returns a list bearing the dataset of predictions from
        our model created.
        
    Usage:
        ols = new OLS(file_name)
        ols.load_data()
        print(ols.get_dataframe())
    """
    def __init__(self,file_name):
        '''
        Constructor for the OLS Class
        file_name(string): The filename of the file bearing the dataset
        to be fitted a linear regression model on using Ordinary Least Square
        Regression method.
        '''
        super().__init__(file_name)
        super().load_data()
        self.dataframe = super().get_dataframe()
        self.model = []
        self.residuals = []
    def fit_regression(self, add_polynomial_term):
        '''
        fit_regression Method for the OLS Class
        returns: The resultant model from the fitting done.
        add_polynomial_term: A boolean list indicating if a polynomial term should be 
        before the fitting in order to capture complexity.
        '''
        #Add polynomial features (degree = 3) to introduce complexity
        degree = 3
        alpha = 0.1 #Regularization parameter
        if add_polynomial_term[0] is True:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            x_poly = poly.fit_transform(np.array(self.dataframe.loc[:,'x']).reshape(-1,1))
            #done to avoid having a zero intercept
            x_with_constant = sm.add_constant(x_poly)  
            model = sm.OLS(self.dataframe.loc[:,'y1'], 
                                        x_with_constant).fit_regularized(alpha=alpha, L1_wt=0.0)
            self.model.insert(0, model)
        else:
            x_with_constant = sm.add_constant(self.dataframe.loc[:,'x'])                     
            self.model.insert(0, sm.OLS(self.dataframe.loc[:,'y1'], 
                                        x_with_constant).fit_regularized(alpha=alpha, L1_wt=0.0))
        if add_polynomial_term[1] is True:
            #Add polynomial features (degree = 10) to introduce complexity
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            x_poly = poly.fit_transform(np.array(self.dataframe.loc[:,'x']).reshape(-1,1))
            #done to avoid having a zero intercept
            x_with_constant = sm.add_constant(x_poly)
            self.model.insert(1, sm.OLS(self.dataframe.loc[:,'y2'], 
                                        x_with_constant).fit_regularized(alpha=alpha, L1_wt=0.0))
        else:
            x_with_constant = sm.add_constant(self.dataframe.loc[:,'x'])                     
            self.model.insert(1, sm.OLS(self.dataframe.loc[:,'y2'], 
                                        x_with_constant).fit_regularized(alpha=alpha, L1_wt=0.0))
        if add_polynomial_term[2] is True:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            x_poly = poly.fit_transform(np.array(self.dataframe.loc[:,'x']).reshape(-1,1))
            #done to avoid having a zero intercept
            x_with_constant = sm.add_constant(x_poly)
            self.model.insert(2, sm.OLS(self.dataframe.loc[:,'y3'], 
                                        x_with_constant).fit_regularized(alpha=alpha, L1_wt=0.0) )
        else:
            x_with_constant = sm.add_constant(self.dataframe.loc[:,'x'])                     
            self.model.insert(2, sm.OLS(self.dataframe.loc[:,'y3'], 
                                        x_with_constant).fit_regularized(alpha=alpha, L1_wt=0.0))
        if add_polynomial_term[3] is True:            
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            x_poly = poly.fit_transform(np.array(self.dataframe.loc[:,'x']).reshape(-1,1))
            #done to avoid having a zero intercept
            x_with_constant = sm.add_constant(x_poly)                     
            self.model.insert(3, sm.OLS(self.dataframe.loc[:,'y4'], 
                                        x_with_constant).fit_regularized(alpha=alpha, L1_wt=0.0))
        else:
            x_with_constant = sm.add_constant(self.dataframe.loc[:,'x'])                     
            self.model.insert(3, sm.OLS(self.dataframe.loc[:,'y4'], 
                                        x_with_constant).fit_regularized(alpha=alpha, L1_wt=0.0))
        return self.model
    def compute_residuals(self, add_polynomial_term):
        '''
        computes the residuals of the regression model created        
        add_polynomial_term: denotes models that polynomial terms
        were added to during the fitting.
        returns: The residuals
        '''
        predicted_values = self.prepare_predicted_value(add_polynomial_term)
        self.residuals.insert(0, np.subtract(self.dataframe.loc[:,'y1'],predicted_values[0]))
        self.residuals.insert(0, np.subtract(self.dataframe.loc[:,'y2'],predicted_values[1]))
        self.residuals.insert(0, np.subtract(self.dataframe.loc[:,'y3'],predicted_values[2]))
        self.residuals.insert(0, np.subtract(self.dataframe.loc[:,'y4'],predicted_values[3]))
        return self.residuals
    def prepare_predicted_value(self, add_polynomial_term):
        '''
        Prepares a 2-D list of predicted values
        returns: a 2-D list with four columns, each column
        bearing the predicted value of y1, y2, y3, and y4
        add_polynomial_term: denotes models that polynomial terms
        were added to during the fitting.
        '''
        predicted_values = []
        i=0
        while i < 4:
            if add_polynomial_term[i] is True:
                poly = PolynomialFeatures(degree=3, include_bias=False)
                x_poly = poly.fit_transform(np.array(self.dataframe.loc[:,'x']).reshape(-1,1))
                #done to avoid having a zero intercept
                x_with_constant = sm.add_constant(x_poly)
                predicted_values.insert(i, self.model[i].predict(x_with_constant))
            else:
                 x_with_constant = sm.add_constant(self.dataframe.loc[:,'x'])
                 predicted_values.insert(i, self.model[i].predict(x_with_constant))
            i = i+1
        dictionary = {"x":self.dataframe.loc[:,'x'],
                      "y1_pred": predicted_values[0],
                      "y2_pred": predicted_values[1],
                      "y3_pred": predicted_values[2],
                      "y4_pred": predicted_values[3]}
        super().write_to_file(pd.DataFrame(data=dictionary), "predictions.csv")
        return predicted_values
        