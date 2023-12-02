'''
Imports modules needed by the class
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
class Visualization(object):
    """
    A simple Visualization class.
    Attributes:
        title(string): The title of the string
        x_label(string): The label for the x axis
        y_label(string): The label for the y axis
    Methods:
        histogram(): Plots a histogram. If you need it to be a variable 
        binned histogram, make the parameter bin to be a list
        of unequal interval between values, with areas you need to
        be concentrated with bins having lesser interval between them.
        scatter_plot(): Creates a scatter plot of y_values 
        against x_values.
        pairplot(): Creates a pairplot of the dataset -
        a matrix of scatter plots
        box_plot(): Creates a box plot showing the statistics
        of the data passed to it.
        line_plot():  Creates a line plot.
        kdeplot(): Creates a kernel density plot
        
        Example:
        visualization = Visualization("box plot", "x values", "y1 function")
        visualization.scatter_plot([1,2,3,4], [2,4,6,8])
            
    """
    def __init__(self, title, x_label, y_label):
        '''
        Constructor for the Visualization Class
        title: The title of the visualization
        x_label: The label for the x axis of the plot.
        y_label: The label for the y axis of the plot.
        '''
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
    def histogram(self, bins, data):
        '''
        Plots a histogram. If you need it to be a variable 
        binned histogram, make the parameter bin to be a list
        of unequal interval between values, with areas you need to
        be concentrated with bins having lesser interval between them.
        bin: The number of bins to distribute the dataset if it a usual
        histogram or a list of values specifying the width of each bin.
        data: The data to be visualized using a histogram.
        '''
        plt.hist(data, bins=bins, density=True, color='Blue', edgecolor='black', alpha=0.7)
        mu, std = norm.fit(data)
        xmin, xmax = plt.xlim()#retrieves the x-axis view limit
        #generates an array of 100 element from xmin to xmax
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x,p,'k',linewidth=2, label="Probability Density Curve")
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.show()
    def scatter_plot(self, x_values, y_values, axhline=False):
        '''
        Creates a scatter plot of y_values 
        against x_values.
        x_values: The list of x values
        y_values: The list of y values
        '''
        plt.scatter(x_values, y_values, label='Scatter Plot', color='blue', marker='o')
        if axhline is True:
            #Horizontal line at y=0
            plt.axhline(y=0, color='black', linestyle='--', linewidth=1) 
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.show()
    def pairplot(self, data, size):
        '''
        Creates a pairplot of the dataset -
        a matrix of scatter plots
        size: The size of the markers in the plot
        '''
        sns.pairplot(data, height=size)
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.title(self.title)
        plt.show()
    def box_plot(self, data):
        '''
        Creates a box plot showing the statistics
        of the data passed to it.
        data: The dataset column to create a box plot
        for.
        '''
        plt.figure(figsize=(6,4))
        sns.boxplot(x=data)
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()
    def line_plot(self, x, y, linewidth):
        '''
        Creates a line plot.
        x: The x axes dataset
        y: The y axes dataset
        linewidth: The width of the line of the
        plot.
        '''
        plt.plot(x,y, label="Line Plot", linewidth=linewidth)
        plt.grid(True, color='k')
        plt.ylabel(self.y_label)
        plt.xlabel(self.x_label)
        plt.title(self.title)
        plt.show()
    def kdeplot(self, data):
        '''
        Creates a kernel density plot with
        the data passed to it.
        data: A list of data
        '''
        plt.figure(figsize=(8,5))
        sns.kdeplot(data, fill=True)
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()
        