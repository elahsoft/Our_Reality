'''
Imports the modules needed by the class
'''
import sqlite3
from utility.dataframe_utility import DataFrameUtility
class Sqlite3Utility(DataFrameUtility):
    """
    A simple Sqlite3Utility Class that loads the extends DataFrameUtility
    class, and writes the dataframe read by DataFrameUtility to sqlite db. 
    Attributes:
        file_name(string): The filename of the file bearing the dataset
        to be loaded.
        dataframe (pandas.dataframe): The dataframe  bearing the dataset to
        be written to a file.
    Methods:
        write(string): Writes the dataframe created from the file read to
        a database table with name as the string passed to it.
            
    Usage:
        sqlite3_utility = new Sqlite3Utility(file_name)
        sqlite3_utility.write('train')
    """
    def __init__(self,file_name):
        '''
        Constructor for the Sqlite3Utility Class
        file_name: The file bearing the dataset for a dataframe
        to be created from.
        '''
        super().__init__(file_name)
        super().load_data()
        self.dataframe = super().get_dataframe()
        self.file_name = file_name
    def write(self, table_name):
        '''
        Writes the dataframe created from the file received by constructor to 
        the database table named 'table_name'
        table_name: A string bearing the name of the databse table to write the 
        dataframe.
        '''
        try:
            conn = sqlite3.connect('our_reality.db')
            self.dataframe.to_sql(table_name, conn, index=False, if_exists='replace')
        except Exception as e:
            print(f"An error occurred: {e}")
            
        finally:
            #Close the connection in the 'finally' block to ensure it's always closed
            if conn:
                conn.close()
