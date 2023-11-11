'''
Imports the modules needed by the class
'''
import statsmodels.api as sm
class OLS(object):
    """
    A simple OLS class.
    Attributes:
        df (pandas.dataframe): The dataframe  bearing the dataset, which we want
        fit a Linear Regression Model over using Ordinary Least Square Method.
    Methods:
        fit_regression(): Fits a linear regression model over the data.
        compute_residuals(): Computes the residuals of the dataset
    """
    def __init__(self, df):
        '''
        Constructor for the OLS Class
        df: The dataframe created from the dataset to be fitted to a
        Linear Regression Model
        '''
        self.dataframe = df
        self.model = []
        self.residuals = []
    def fit_regression(self):
        '''
        fit_regression Method for the OLS Class
        returns: The resultant model from the fitting done.
        '''
        #done to avoid having a zero intercept
        x_with_constant = sm.add_constant(self.dataframe.loc[:,'x']) 
        self.model.insert(0, sm.OLS(self.dataframe.loc[:,'y1'], x_with_constant).fit())
        self.model.insert(1, sm.OLS(self.dataframe.loc[:,'y2'], x_with_constant).fit())
        self.model.insert(2, sm.OLS(self.dataframe.loc[:,'y3'], x_with_constant).fit())
        self.model.insert(3, sm.OLS(self.dataframe.loc[:,'y4'], x_with_constant).fit())
        return self.model
    def compute_residuals(self):
        '''
        computes the residuals of the regression model created
        returns: The residuals
        '''
        self.residuals.insert(0, self.model[0].resid)
        self.residuals.insert(1, self.model[1].resid)
        self.residuals.insert(2, self.model[2].resid)
        self.residuals.insert(3, self.model[3].resid)
        return self.residuals
        