'''
Imports the modules needed by the class
'''
import statsmodels.api as sm
import pandas as pd
import config.config as cf
class OLS(object):
    """
    A simple OLS class.
    Attributes:
        file_name(string): The filename of the file bearing the dataset
        to be fitted a linear regression model on using Ordinary Least Square
        Regression method.
        df (pandas.dataframe): The dataframe  bearing the dataset, which we want
        fit a Linear Regression Model over using Ordinary Least Square Method.
        model(list): A list of models created for y1, y2, y3 and y4
        residuals(list): A  list bearing the residuals of y1, y2, y3, y4 with 
        regards to the training dataset used in creating the model
    Methods:
        load_data(): Loads the dataset from the file into a pandas dataframe.
        get_dataframe(): gets the dataframe created with the file passed to the class
        fit_regression(): Fits a linear regression model over the data.
        compute_residuals(): Computes the residuals of the dataset
        prepare_predicted_value(): Returns a list bearing the dataset of predictions from
        our model created.
    """
    def __init__(self,file_name):
        '''
        Constructor for the OLS Class
        df: The dataframe created from the dataset to be fitted to a
        Linear Regression Model
        '''
        self.dataframe = None
        self.file_name = file_name
        self.model = []
        self.residuals = []
    def load_data(self):
        '''
        Loads the file received by constructor to pandas dataframe
        '''
        df_data = None
        try:
            df_data = pd.read_csv(filepath_or_buffer=cf.INPUT_FILE_PATH+self.file_name,
                                  sep=",", encoding="latin1")
        except FileNotFoundError:
            print(f"Error: The file '{self.file_name}' was not found")
        except pd.errors.EmptyDataError:
            print(f"Error: The file '{self.file_name}' is empty")
        except pd.errors.ParserError as e:
            print(f"Error while parsing csv: {e}")
        except Exception as e:
            print(f"An unexpected error occurred {e}")
        finally:
            pass
        self.dataframe = df_data
    def get_dataframe(self):
        '''
        gets the dataframe created with the file passed to the class
        returns: A dataframe
        '''
        return self.dataframe
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
    def prepare_predicted_value(self):
        '''
        Prepares a 2-D list of predicted values
        returns: a 2-D list with four columns, each column
        bearing the predicted value of y1, y2, y3, and y4
        '''
        predicted_values = [None, None, None, None]
        x_with_constant = sm.add_constant(self.dataframe.loc[:,'x'])
        i=0
        while i < 4:
            predicted_values[i] = self.model[i].predict(x_with_constant)
            i = i+1
        return predicted_values
        