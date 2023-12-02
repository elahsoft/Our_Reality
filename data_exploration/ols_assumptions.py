'''
Imports modules needed by the class
'''
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import linear_rainbow
import copy
from data_exploration.ols import OLS
from utility.dataframe_utility import DataFrameUtility
class OLSAssumptions(OLS):
    """
    A simple OLSAssumptions class. It is a child class of OLS.
    Attributes:
        file_name (string): The file name of the file bearing the dataset, which we want
        to test if it fulfills all the characteristics of a dataset that can be modelled
        using an Ordinary Least Square Method.
        self.model(list) A list bearing the models we created via the fitting
        self.residuals(list): A list bearing the residuals of the model created with 
        regards to the training dataset used in creating it.
        dw_statistics(list): A list bearing the float values being the  result of
        durbin-watson test carried out on the training dataset.
        p_value(list): A list bearing the float values being the  result of
        durbin-watson test carried out on the training dataset. 
        jb_p_value(list): A list bearing the float values being the  result of
        Jarque-Bera test carried out on the training dataset.
        rainbow_p_value(list): A list bearing the float values being the  result of
        Rainbow test carried out on the training dataset.
    Methods:
        fit_regression(): Fits a linear regression model over the data.
        check_heterocedasticity(): Uses Breusch-Pagan Test to check for
        the heterocedasticity of the dataset.
        check_normality(): Uses Jarque-Bera Test to check for the normality of the residuals.
        check_linearity(): Uses a scatter plot to show the linearity, visual inspection
        is needed to arrive at a decision about this feature.
        check_independence(): Uses Durbin-Watson Statistics to test for independence
        
        Example:
            ols_assumptions = OLSAssumptions(dataframe)
            linearity_status = ols_assumptions.check_linearity()
            print(linearity_status)  # Output:
    """
    def __init__(self, file_name=None, add_polynomial_term=None):
        '''
        Constructor for the OLSAssumptions Class
        file_name: The file_name of the file bearing the dataset to be checked for it's
        fulfillment of OLS assumptions.
        add_polynomial_term: A boolean list indicating if a polynomial term should be 
        before the fitting in order to capture complexity.
        '''
        if file_name is not None and add_polynomial_term is not None:
            super().__init__(file_name)
            super().load_data()
            self.model =  super().fit_regression(add_polynomial_term)
            self.predicted_value = super().prepare_predicted_value(add_polynomial_term)
            self.residuals = super().compute_residuals(add_polynomial_term)
            self.dataframe = super().get_dataframe()
        self.dw_statistics = None
        self.p_value = None
        self.jb_p_value = None
        self.rainbow_p_value = None
    def check_heterocedasticity(self, fit=True):
        '''
        Uses Breusch-Pagan Test to check for the heterocedasticity of the dataset.
        fit: Indicates if a fitting was done or we read the residuals from the file.
        returns: A Boolean list indicating the heterocedascity of the residuals
        '''
        if fit is True:
            _, p_value0,_,_ = het_breuschpagan(self.residuals[0],
                                            sm.add_constant(self.dataframe.loc[:,'x']))
            _, p_value1,_,_ = het_breuschpagan(self.residuals[1],
                                            sm.add_constant(self.dataframe.loc[:,'x']))
            _, p_value2,_,_ = het_breuschpagan(self.residuals[2],
                                            sm.add_constant(self.dataframe.loc[:,'x']))
            _, p_value3,_,_ = het_breuschpagan(self.residuals[3],
                                            sm.add_constant(self.dataframe.loc[:,'x']))
        else:
            dataframe_utility = DataFrameUtility('residuals.csv')
            dataframe_utility.load_data()
            res_df = dataframe_utility.dataframe
            dataframe_utility = DataFrameUtility('train.csv')
            dataframe_utility.load_data()
            train_df = dataframe_utility.dataframe
            _, p_value0,_,_ = het_breuschpagan(res_df.loc[:,'res1'],
                                            sm.add_constant(train_df.loc[:,'x']))
            _, p_value1,_,_ = het_breuschpagan(res_df.loc[:,'res2'],
                                            sm.add_constant(train_df.loc[:,'x']))
            _, p_value2,_,_ = het_breuschpagan(res_df.loc[:,'res3'],
                                            sm.add_constant(train_df.loc[:,'x']))
            _, p_value3,_,_ = het_breuschpagan(res_df.loc[:,'res4'],
                                            sm.add_constant(train_df.loc[:,'x']))
            
            self.p_value = [p_value0, p_value1, p_value2, p_value3]
        result = [False, False, False, False] #Is Homocedastic and so no transformation needed
        if p_value0 < 0.05:
            result.pop(0)
            result.insert(0, True) #Is Not Homocedastic and so transformation needed for y
        if p_value1 < 0.05:
            result.pop(1)
            result.insert(1, True) #Is Not Homocedastic and so transformation needed for y
        if p_value2 < 0.05:
            result.pop(2)
            result.insert(2, True) #Is Not Homocedastic and so transformation needed for y
        if p_value3 < 0.05:
            result.pop(3)
            result.insert(3, True) #Is Not Homocedastic and so transformation needed for y

        return result
    def check_normality(self, fit=True):
        '''
        Uses Jarque-Bera Test to check for the Normality of the dataset.
        It also uses a Q-Q plot to visually compare the distribution
        of the residuals against a normal distribution. A straight line in
        the plot suggests normality.        
        fit: Indicates if a fitting was done or we read the residuals from the file.
        '''
        #plot to show normality of residuals
        if fit is True:
            res = self.residuals
        else:
            dataframe_utility = DataFrameUtility('residuals.csv')
            dataframe_utility.load_data()
            res = [dataframe_utility.dataframe.loc[:,'res1'],
                    dataframe_utility.dataframe.loc[:,'res2'],
                    dataframe_utility.dataframe.loc[:,'res3'],
                    dataframe_utility.dataframe.loc[:,'res4']]
        _, axes = plt.subplots(nrows=2, ncols=2,
                                 figsize=(10,8)) #creates a 2rows and 2columns subplots
        #Y1 Residuals Q-Q Plot
        sm.qqplot(res[0], line='s', ax=axes[0,0])
        axes[0,0].set_title('Y1 Residuals Q-Q Plot')
        axes[0,0].legend()
        #Y2 Residuals Q-Q Plot
        sm.qqplot(res[1], line='s', ax=axes[0,1])
        axes[0,1].set_title('Y2 Residuals Q-Q Plot')
        axes[0,1].legend()
        #Y3 Residuals Q-Q Plot
        sm.qqplot(res[2], line='s', ax=axes[1,0])
        axes[1,0].set_title('Y3 Residuals Q-Q Plot')
        axes[1,0].legend()
        #Y4 Residuals Q-Q Plot
        sm.qqplot(res[3], line='s', ax=axes[1,1])
        axes[1,1].set_title('Y4 Residuals Q-Q Plot')
        axes[1,1].legend()
        #Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()       
        _,jb_p_value1,_,_ = sm.stats.stattools.jarque_bera(res[0])
        _,jb_p_value2,_,_ = sm.stats.stattools.jarque_bera(res[1])
        _,jb_p_value3,_,_ = sm.stats.stattools.jarque_bera(res[2])
        _,jb_p_value4,_,_ = sm.stats.stattools.jarque_bera(res[3])
        self.jb_p_value = [jb_p_value1, jb_p_value2, jb_p_value3, jb_p_value4]
        #Initial assumption of non-normality of residuals
        result = [False, False, False, False]
        if jb_p_value1 > 0.05 or jb_p_value1 == 0.05:
            result.pop(0)
            result.insert(0,True)
        if jb_p_value2 > 0.05 or jb_p_value2 == 0.05:
            result.pop(1)
            result.insert(1,True)
        if jb_p_value3 > 0.05 or jb_p_value3 == 0.05:
            result.pop(2)
            result.insert(2,True)
        if jb_p_value4 > 0.05 or jb_p_value4 == 0.05:
            result.pop(3)
            result.insert(3,True)
        return result
    def check_linearity(self, fit=True):
        '''
        Uses a scatter plot to show the linearity, visual inspection
        is needed to arrive at a decision about this feature.
        No pattern should be observed in the plot. If patterns are
        observed, it is an indication of violation of linear
        regression assumption of linearity, so we can't fit a 
        linear regression model on the dataset. It also uses a 
        rainbow test to further check for linearity assumption        
        fit: Indicates if a fitting was done or we read the residuals from the file.
        returns: A list with boolean values indicating linearity
        '''
        if fit is True:
            res = copy.deepcopy(self.residuals)
            pred = [self.model[0].fittedvalues,
                    self.model[1].fittedvalues,
                    self.model[2].fittedvalues,
                    self.model[3].fittedvalues]
        else:
            dataframe_utility = DataFrameUtility('residuals.csv')
            dataframe_utility.load_data()
            res = [dataframe_utility.dataframe.loc[:,'res1'],
                    dataframe_utility.dataframe.loc[:,'res2'],
                    dataframe_utility.dataframe.loc[:,'res3'],
                    dataframe_utility.dataframe.loc[:,'res4']]
            dataframe_utility = DataFrameUtility('predictions.csv')
            dataframe_utility.load_data()
            pred = [dataframe_utility.dataframe.loc[:,'y1'],
                    dataframe_utility.dataframe.loc[:,'y2'],
                    dataframe_utility.dataframe.loc[:,'y3'],
                    dataframe_utility.dataframe.loc[:,'y4']] 
        #Check for linearity by plotting Residuals Against Fitted Values
        _, axes = plt.subplots(nrows=2, ncols=2,
                                 figsize=(10,8)) #creates a 2rows and 2columns subplots
        #Y1 Scatter Plot of Residuals Against Fitted Values
        axes[0,0].scatter(pred[0], res[0], color='red',
                          label="Y1 Residuals Against Fitted Values")
        #Horizontal line at y=0
        axes[0,0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[0,0].set_title('Y1 Scatter Plot of Residuals Against Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        axes[0,0].legend()
        #Y2 Scatter Plot of Residuals Against Fitted Values
        axes[0,1].scatter(pred[1], res[1], color='blue',
                          label="Y2 Residuals Against Fitted Values")
        #Horizontal line at y=0
        axes[0,1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[0,1].set_title('Y2 Scatter Plot of Residuals Against Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        axes[0,1].legend()
        #Y3 Scatter Plot of Residuals Against Fitted Values
        axes[1,0].scatter(pred[2], res[2], color='green',
                          label="Y3 Residuals Against Fitted Values")
        #Horizontal line at y=0
        axes[1,0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1,0].set_title('Y3 Scatter Plot of Residuals Against Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        axes[1,0].legend()
        #Y4 Scatter Plot of Residuals Against Fitted Values
        axes[1,1].scatter(pred[3], res[3], color='purple',
                          label="Y4 Residuals Against Fitted Values")
        #Horizontal line at y=0
        axes[1,1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1,1].set_title('Y4 Scatter Plot of Residuals Against Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        axes[1,1].legend()
        #Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
        result = [True, True, True, True] #Linearity assumed initially
        if fit is True:
            try:
                #Use rainbow test to further check for linearity assumption
                _,rainbow_p_value1 = linear_rainbow(self.model[0])
                _,rainbow_p_value2 = linear_rainbow(self.model[1])
                _,rainbow_p_value3 = linear_rainbow(self.model[2])
                _,rainbow_p_value4 = linear_rainbow(self.model[3])
                self.rainbow_p_value = [rainbow_p_value1, rainbow_p_value2,
                                        rainbow_p_value3, rainbow_p_value4]
                #Indication of Non-linearity, so we may try transforming variables
                if rainbow_p_value1 < 0.05:
                    result.pop(0)
                    result.insert(0, False)
                #Indication of Non-linearity, so we may try transforming variables
                if rainbow_p_value2 < 0.05:
                    result.pop(0)
                    result.insert(0, False)
                #Indication of Non-linearity, so we may try transforming variables
                if rainbow_p_value3 < 0.05:
                    result.pop(0)
                    result.insert(0, False)
                #Indication of Non-linearity, so we may try transforming variables
                if rainbow_p_value4 < 0.05:
                    result.pop(0)
                    result.insert(0, False)
            except TypeError as te:
                print(f"An unexpected error occurred {te}")        
        return result
    def check_independence(self, fit=True):
        '''
        Uses Durbin-Watson Statistics to test for independence
        and also a scatter plot of residuals against order of observation
        returns: A Boolean list indicating the autocorrelation or serial correlation in
        the residuals, values True indicating independence, i.e. no autocorrelation 
        of residuals, False meaning the opposite.        
        fit: Indicates if a fitting was done or we read the residuals from the file.
        '''
        if fit is True:
            res = self.residuals
        else:
            dataframe_utility = DataFrameUtility('residuals.csv')
            dataframe_utility.load_data()
            res = [dataframe_utility.dataframe.loc[:,'res1'],
                    dataframe_utility.dataframe.loc[:,'res2'],
                    dataframe_utility.dataframe.loc[:,'res3'],
                    dataframe_utility.dataframe.loc[:,'res4']]
        dw_statistics1 = durbin_watson(res[0])
        dw_statistics2 = durbin_watson(res[1])
        dw_statistics3 = durbin_watson(res[2])
        dw_statistics4 = durbin_watson(res[3])
        self.dw_statistics = [dw_statistics1, dw_statistics2,
                              dw_statistics3, dw_statistics4]
        result = [False, False, False, False] #Assume autocorrelation at the beginning
        if dw_statistics1 >= 1.5 and dw_statistics1 <= 2.5:
            result.pop(0)
            result.insert(0, True)
        if dw_statistics2 >= 1.5 and dw_statistics2 <= 2.5:
            result.pop(1)
            result.insert(1, True)
        if dw_statistics3 >= 1.5 and dw_statistics3 <= 2.5:
            result.pop(2)
            result.insert(2, True)
        if dw_statistics4 >= 1.5 and dw_statistics4 <= 2.5:
            result.pop(3)
            result.insert(3, True)
        #Check for independence of residuals i.e. no autocorrelation
        # by plotting residuals against order of observation
        _, axes = plt.subplots(nrows=2, ncols=2,
                                 figsize=(10,8)) #creates a 2rows and 2columns subplots
        #Y1 Scatter Plot of Residuals Against Order of Observation
        axes[0,0].scatter(range(len(res[0])), res[0], color='red',
                          label="Y1 Residuals Against order of observation")
        axes[0,0].set_title('Y1 Scatter Plot of Residuals Against order of observation')
        plt.xlabel('Order of Observation')
        plt.ylabel('Residuals')
        axes[0,0].legend()
        #Y2 Scatter Plot of Residuals Against Order of Observation
        axes[0,1].scatter(range(len(res[1])), res[1], color='blue',
                          label="Y2 Residuals Against order of observation")
        axes[0,1].set_title('Y2 Scatter Plot of Residuals Against order of observation  ')
        plt.xlabel('Order of Observation')
        plt.ylabel('Residuals')
        axes[0,1].legend()
        #Y3 Scatter Plot of Residuals Against Order of Observation
        axes[1,0].scatter(range(len(res[2])), res[2], color='green',
                          label="Y3 Residuals Against order of observation")
        axes[1,0].set_title('Y3 Scatter Plot of Residuals Against order of observation')
        plt.xlabel('Order of Observation')
        plt.ylabel('Residuals')
        axes[1,0].legend()
        #Y4 Scatter Plot of Residuals Against Order of Observation
        axes[1,1].scatter(range(len(res[3])), res[3], color='purple',
                          label="Y4 Residuals Against order of observation")
        axes[1,1].set_title('Y4 Scatter Plot of Residuals Against order of observation')
        plt.xlabel('Order of Observation')
        plt.ylabel('Residuals')
        axes[1,1].legend()
        #Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show() 
        return result
        