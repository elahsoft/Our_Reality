'''
Imports modules needed by the class
'''
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import linear_rainbow
import scipy.stats as stats
from data_exploration.ols import OLS
class OLSAssumptions(object):
    """
    A simple OLSAssumptions class.
    Attributes:
        df (pandas.dataframe): The dataframe bearing the dataset, which we want
        to test if it fulfills all the characteristics of a dataset that can be modelled
        using an Ordinary Least Square Method.
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
    def __init__(self, df):
        '''
        Constructor for the OLSAssumptions Class
        df: The dataframe created from the dataset to be checked for it's
        fulfillment of OLS assumptions.
        '''
        self.dataframe = df
        ols = OLS(df)
        self.model =  ols.fit_regression()
        self.residuals = ols.compute_residuals()
        self.dw_statistics = None
        self.p_value = None
        self.jb_p_value = None
        self.rainbow_p_value = None
    def check_heterocedasticity(self):
        '''
        Uses Breusch-Pagan Test to check for the heterocedasticity of the dataset.
        returns: A Boolean list indicating the heterocedascity of the residuals
        '''
        _, p_value0,_,_ = het_breuschpagan(self.residuals[0],
                                           sm.add_constant(self.dataframe.loc[:,'x']))
        _, p_value1,_,_ = het_breuschpagan(self.residuals[1],
                                           sm.add_constant(self.dataframe.loc[:,'x']))
        _, p_value2,_,_ = het_breuschpagan(self.residuals[2],
                                           sm.add_constant(self.dataframe.loc[:,'x']))
        _, p_value3,_,_ = het_breuschpagan(self.residuals[3],
                                           sm.add_constant(self.dataframe.loc[:,'x']))
        self.p_value = [p_value0, p_value1, p_value2, p_value3]
        print("het_breu_p_value", self.p_value)
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
    def check_normality(self):
        '''
        Uses Jarque-Bera Test to check for the Normality of the dataset.
        It also uses a Q-Q plot to visually compare the distribution
        of the residuals against a normal distribution. A straight line in
        the plot suggests normality.
        returns: A Boolean indicating the normality of the residuals
        '''
        #plot to show normality of residuals
        _, axes = plt.subplots(nrows=2, ncols=2,
                                 figsize=(10,8)) #creates a 2rows and 2columns subplots
        #Y1 Residuals Q-Q Plot
        sm.qqplot(self.residuals[0], line='s', ax=axes[0,0])
        axes[0,0].set_title('Y1 Residuals Q-Q Plot')
        axes[0,0].legend()
        #Y2 Residuals Q-Q Plot
        sm.qqplot(self.residuals[1], line='s', ax=axes[0,1])
        axes[0,1].set_title('Y2 Residuals Q-Q Plot')
        axes[0,1].legend()
        #Y3 Residuals Q-Q Plot
        sm.qqplot(self.residuals[2], line='s', ax=axes[1,0])
        axes[1,0].set_title('Y3 Residuals Q-Q Plot')
        axes[1,0].legend()
        #Y4 Residuals Q-Q Plot
        sm.qqplot(self.residuals[3], line='s', ax=axes[1,1])
        axes[1,1].set_title('Y4 Residuals Q-Q Plot')
        axes[1,1].legend()
        #Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()       
        _,jb_p_value1,_,_ = sm.stats.stattools.jarque_bera(self.residuals[0])
        _,jb_p_value2,_,_ = sm.stats.stattools.jarque_bera(self.residuals[1])
        _,jb_p_value3,_,_ = sm.stats.stattools.jarque_bera(self.residuals[2])
        _,jb_p_value4,_,_ = sm.stats.stattools.jarque_bera(self.residuals[3])
        self.jb_p_value = [jb_p_value1, jb_p_value2, jb_p_value3, jb_p_value4]
        print("jb_p_value", self.jb_p_value)
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
    def check_linearity(self):
        '''
        Uses a scatter plot to show the linearity, visual inspection
        is needed to arrive at a decision about this feature.
        No pattern should be observed in the plot. If patterns are
        observed, it is an indication of violation of linear
        regression assumption of linearity, so we can't fit a 
        linear regression model on the dataset. It also uses a 
        rainbow test to further check for linearity assumption
        returns: A list with boolean values indicating linearity
        '''
        #Check for linearity by plotting Residuals Against Fitted Values
        _, axes = plt.subplots(nrows=2, ncols=2,
                                 figsize=(10,8)) #creates a 2rows and 2columns subplots
        #Y1 Scatter Plot of Residuals Against Fitted Values
        axes[0,0].scatter(self.model[0].fittedvalues, self.residuals[0], color='red',
                          label="Y1 Residuals Against Fitted Values")
        axes[0,0].set_title('Y1 Scatter Plot of Residuals Against Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        axes[0,0].legend()
        #Y2 Scatter Plot of Residuals Against Fitted Values
        axes[0,1].scatter(self.model[1].fittedvalues, self.residuals[1], color='blue',
                          label="Y2 Residuals Against Fitted Values")
        axes[0,1].set_title('Y2 Scatter Plot of Residuals Against Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        axes[0,1].legend()
        #Y3 Scatter Plot of Residuals Against Fitted Values
        axes[1,0].scatter(self.model[2].fittedvalues, self.residuals[2], color='green',
                          label="Y3 Residuals Against Fitted Values")
        axes[1,0].set_title('Y3 Scatter Plot of Residuals Against Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        axes[1,0].legend()
        #Y4 Scatter Plot of Residuals Against Fitted Values
        axes[1,1].scatter(self.model[3].fittedvalues, self.residuals[3], color='purple',
                          label="Y4 Residuals Against Fitted Values")
        axes[1,1].set_title('Y4 Scatter Plot of Residuals Against Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        axes[1,1].legend()
        #Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()       
        #Use rainbow test to further check for linearity assumption
        _,rainbow_p_value1 = linear_rainbow(self.model[0])
        _,rainbow_p_value2 = linear_rainbow(self.model[1])
        _,rainbow_p_value3 = linear_rainbow(self.model[2])
        _,rainbow_p_value4 = linear_rainbow(self.model[3])
        self.rainbow_p_value = [rainbow_p_value1, rainbow_p_value2,
                                rainbow_p_value3, rainbow_p_value4]
        print("rainbow_p_value", self.rainbow_p_value)
        result = [True, True, True, True] #Linearity assumed initially
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
        return result
    def check_independence(self):
        '''
        Uses Durbin-Watson Statistics to test for independence
        and also a scatter plot of residuals against order of observation
        returns: A Boolean list indicating the autocorrelation or serial correlation in
        the residuals, values True indicating independence, i.e. no autocorrelation 
        of residuals, False meaning the opposite.
        '''
        dw_statistics1 = durbin_watson(self.residuals[0])
        dw_statistics2 = durbin_watson(self.residuals[1])
        dw_statistics3 = durbin_watson(self.residuals[2])
        dw_statistics4 = durbin_watson(self.residuals[3])
        self.dw_statistics = [dw_statistics1, dw_statistics2,
                              dw_statistics3, dw_statistics4]
        result = [False, False, False, False] #Assume autocorrelation at the beginning
        
        print("dw_statistics1 ", self.dw_statistics)
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
        # by plotting order of observation Against residuals
        _, axes = plt.subplots(nrows=2, ncols=2,
                                 figsize=(10,8)) #creates a 2rows and 2columns subplots
        #Y1 Scatter Plot of Residuals Against Fitted Values
        axes[0,0].scatter(range(len(self.residuals[0])), self.residuals[0], color='red',
                          label="Y1 order of observation Against Residuals")
        axes[0,0].set_title('Y1 Scatter Plot of order of observation Against Residuals')
        plt.xlabel('Order of Observation')
        plt.ylabel('Residuals')
        axes[0,0].legend()
        #Y2 Scatter Plot of Residuals Against Fitted Values
        axes[0,1].scatter(range(len(self.residuals[1])), self.residuals[1], color='blue',
                          label="Y2 order of observation Against Residuals")
        axes[0,1].set_title('Y2 Scatter Plot of order of observation Against Residuals')
        plt.xlabel('Order of Observation')
        plt.ylabel('Residuals')
        axes[0,1].legend()
        #Y3 Scatter Plot of Residuals Against Fitted Values
        axes[1,0].scatter(range(len(self.residuals[2])), self.residuals[2], color='green',
                          label="Y3 order of observation Against Residuals")
        axes[1,0].set_title('Y3 Scatter Plot of order of observation Against Residuals')
        plt.xlabel('Order of Observation')
        plt.ylabel('Residuals')
        axes[1,0].legend()
        #Y4 Scatter Plot of Residuals Against Fitted Values
        axes[1,1].scatter(range(len(self.residuals[0])), self.residuals[0], color='purple',
                          label="Y4 order of observation Against Residuals")
        axes[1,1].set_title('Y4 Scatter Plot of order of observation Against Residuals')
        plt.xlabel('Order of Observation')
        plt.ylabel('Residuals')
        axes[1,1].legend()
        #Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()      
        return result
        