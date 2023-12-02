'''
Imports the modules needed by the class
'''
import os
import pandas as pd
import config.config as cf
class DataFrameUtility(object):
    """
    A simple DataFrameUtility Class that loads the file bearing the dataset and 
    produce a dataframe.
    Attributes:
        file_name(string): The filename of the file bearing the dataset
        to be loaded.
        dataframe (pandas.dataframe): The dataframe  bearing the dataset to
        be written to a file.
    Methods:
        load_data(): Loads the dataset from the file into a pandas dataframe.
        get_dataframe(): gets the dataframe created with the file passed to the class
        write_to_file(pandas.dataframe, string): write the dataframe to file named as
        the string passed to it.
            
    Usage:
        dataframe_utility = new DataFrameUtility(file_name)
        dataframe_utility.load_data()
        print(dataframe_utility.get_dataframe().head())
    """
    def __init__(self,file_name=None):
        '''
        Constructor for the DataFrameUtility Class
        file_name: The file bearing the dataset for a dataframe
        to be created from.
        '''
        self.dataframe = None
        self.file_name = file_name
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
    def write_to_file(self, df, output_file):
        '''
        write the dataframe to file 
        df: The dataframe to be written to file
        output_file: The file name of the output file
        '''
        df.to_csv(os.path.join(cf.INPUT_FILE_PATH, output_file), index=False)
