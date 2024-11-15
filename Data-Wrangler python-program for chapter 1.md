**Full name:** Lesedi Kopeledi Matshehla  
    
**Description of the Task:** Python-program that uses training data to choose the four ideal
functions which are the best fit out of the fifty provided (C) *. You get (A) 4 training datasets and
(B) one test dataset, as well as (C) datasets for 50 ideal functions. All data respectively consists of
x-y-pairs of values.


```python
**Data-Wrangler python-program for chapter 1.**
```


```python
**Firstly, install all of the required packages to execute the code successfully.**
```


```python
pip install config
```


```python
pip install data-wrangler
```


```python
pip install Pandas-Data-Exploration-Utility-Package
```


```python
pip list
```


```python
**Restart Kernel to activate the list of installed packages.**
```


```python
**Read through the program and comments. Edit the view and layout of the code to execute without errors.**
```


```python
Imports all the modules needed by the class.
'''
import multiprocessing as mp
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
#import config.config as cf
import config as cf
class DataWrangler(object):
#class DataWrangler:
    """
    A simple DataWrangler class.
    Attributes:
        file_name (str): The name of the file bearing our dataset to be wrangled.
    Methods:
        load_data(): Loads the dataset from the file into a pandas dataframe.
        shape_of_data(df_data): Returns the shape of the dataframe passed to it.
        summary_statistics(column_data): Returns a summary statistics of pandas dataframe 
        column passed to it.
        find_missing_values(df_data): Returns a list of missing values (nan) by columns.
        duplicated_rows(df_data): Returns duplicated rows
        drop_duplicates(df_data): Returns a pandas dataframe with duplicated rows dropped.
        fill_missing_values(df_data): Fills missing values (nan) in the dataframe with the mean 
        value of each column
        find_outliers(df_column): Returns all values that are outliers in the numeric 
        column passed to it.
        is_outlier(value): Returns the value if it is an outlier
        fill_outliers_with_mean(outliers, df_column): fills all entries in the dataframe 
        column passed to it, that are outliers with the mean value of the column.
        sort_data(df_data): Sorts the dataframe by the 'x' column    

        Example:
            data_wrangler = DataWrangler(file_name="train.csv")
            df_data = data_wrangler.load_data()
            print(df_data.head(5))  # Output: The first five rows of the train dataset
    """
    def __init__(self, file_name):
        '''
        Constructor for the DataWrangler Class
        file_name: The Name of the file from which we get our raw data for wrangling
        '''
        self.file_name = file_name
        self.lower_bound_outlier = 0
        self.upper_bound_outlier = 0
    def load_data(self):
        '''
        Loads the file received by constructor to pandas dataframe
        return: A pandas dataframe bearing the content of our file passed to constructor
        '''
        try:
            #df_data = pd.read_csv(self.file_name)
            df_data = pd.read_csv(filepath_or_buffer=cf.INPUT_FILE_PATH+self.file_name, 
                            sep=",", encoding="latin1")
        except:
            exception_type, exception_value, exception_traceback = sys.exc.info()
            print("Exception Type: {}\n Exception Value:{}".format(
                exception_type, exception_value))
            file_name, line_number, procedure_name, line_code = traceback.extract_tb(
                exception_traceback)[-1]
            print("File Name: {}\n Line Number:{} \n Procedure Name: {} \n"+
                "Line Code: {}".format(file_name, line_number, procedure_name,
                                        line_code))
        finally:
            pass
        return df_data
    def shape_of_data(self, df_data):
        '''
        Computes the shape of the dataframe passed to it
        return: A tuple, which is the shape of the dataframe.
        '''
        df_shape = df_data.shape
        return df_shape
    def summary_statistics(self, column_data):
        '''
        Computes the summary statistic of each column of the dataframe
        return: A dataframe, which gives a summary statistic of each
        column in the dataframe.
        '''
        return column_data.describe()
    def find_missing_values(self, df_data):
        '''
        Computes the count of missing values in each column of the dataframe
        return: Returns a pandas series bearing the details on the count of missing
        values in each column of the dataframe
        '''
        df_result = df_data.apply(lambda x: sum(x.isnull()), axis=0)
        return df_result
    def duplicated_rows(df_data):
        '''
        Finds all duplicated rows in the dataframe
        return: Returns a dataframe bearing duplicated rows in the dataframe
        '''
        df_duplicated = df_data.duplicated()
        return df_duplicated
    def drop_duplicated(self, df_data):
        '''
        Drops all duplicated rows in the dataframe and keeps the first
        occurrence
        return: Returns a dataframe bearing no duplicated rows
        '''
        df_data = df_data.drop_duplicates(keep="first")
        return df_data
    def fill_missing_values(self, df_data):
        '''
        Fills all missing values in each column with the mean value of the
        column
        return: Returns a dataframe bearing no missing value
        '''
        df_data = df_data.fillna(df_data['x':'y4'].mean())
        return df_data
    def find_outliers(self, df_column):
        '''
        Finds all outlier values in the dataframe column passed to it
        return: Returns a list bearing all outlier values in the column
        '''
        q3 = np.percentile(df_column, 75)
        q1 = np.percentile(df_column, 25)
        iqr = q3-q1
        # Computes the outlier upper and lower bound values
        self.lower_bound_outlier = q1 - (1.5 * iqr)
        self.upper_bound_outlier = q3 + (1.5 * iqr)
        pool_obj = mp.Pool()
        outliers = pool_obj.map(is_outlier, df_column)
        return outliers
    def is_outlier(self, value):
        '''
        Determines if a value is an outlier based on our 
        dataset column in consideration
        return: The Numeric Value passed to it is less than the computed
        lower bound outlier determinant or greater than the upper bound
        outlier determinant
        '''
        if value < self.lower_bound_outlier:
            return value
        if value > self.upper_bound_outlier:
            return value
    def fill_outliers_with_mean(self, outliers, df_column):
        '''
        Replaces all outlier values in the dataframe column passed to it with
        the mean of the values in the column
        return: The Dataframe column with outliers replaced by the mean value
        '''
        mean = df_column.mean()
        new_value = {}
        for value in outliers:
            new_value.update({value:mean})
            df_column.replace(to_replace=new_value, inplace=True)
        return df_column
    def sort_data(self, df_data):
        '''
        Sorts the dataframe passed to it by the column x
        return: The sorted dataframe
        '''
        sorted_df_data = df_data.sort_values(by='x')
        return sorted_df_data
print("Data Wrangler program executed without errors.")

import pandas_exploration_util.viz.explore as pe
#from DW import DataWrangler
#from data_exploration.data_wrangler import DataWrangler
if __name__ == "__main__":
    #unittest.main()
    # Please change file name and uncomment some part of the program to execute ideal, train and test files, respectively.
    print("load_data(): Loads the dataset from the file into a pandas dataframe. Remember to change file name.")
    file_name = "test.csv"
    dataWrangler = DataWrangler(file_name)
    df_data = pd.read_csv("test.csv")
    
    print(df_data)
    print("")
    print("The compressed view of dataset file to do a quick check of your DataFrame.")
    display(df_data)
    print("")
    
    print("Return: A tuple, which is the shape of the dataframe.")
    df_shape= dataWrangler.shape_of_data(df_data)
    print(df_shape)
    
    print("")
    print("Computes the summary statistic of each column of the dataframe.")
    print("Return: A dataframe, which gives a summary statistic of each column in the dataframe.")
    column_data = dataWrangler.summary_statistics(df_data)
    print(column_data)
    print("")
    display(column_data)
    
    print("")
    print("Computes the count of missing values in each column of the dataframe.")
    print("Return: Returns a pandas series bearing the details on the count of missing values in each column of the dataframe.")
    df_result= dataWrangler.find_missing_values(df_data)
    print(df_result)
    print("")
    display(df_result)
    
    print("")
    print("Finds all duplicated rows in the dataframe.")
    print("Return: Returns a dataframe bearing duplicated rows in the dataframe.")
    print("")
    #df_duplicated= dataWrangler.duplicated_rows()
    df_duplicated = df_data.duplicated()
    print(df_duplicated)
    print("")
    display(df_result)
    

    print("")
    print("Drops all duplicated rows in the dataframe and keeps the first occurrence.")
    print("Return: Returns a dataframe bearing no duplicated rows")
    print("")
    #df_data = dataWrangler.drop_duplicated()
    df_data = df_data.drop_duplicates(keep="first")
    print(df_data)
    
#     print("")
#     print("Returns a dataframe bearing no missing value.")
#     #df_data= dataWrangler.fill_missing_values(df_data)
#     #df_data = df_data.fillna(df_data['x':'y4'].mean())
#     print(df_data)

    #Sort values in each column
    sorted_df_data = df_data.apply(lambda s: s.sort_values().values)
    print("\nDataFrame with sorted values in each column:\n")
    print(sorted_df_data)
    
    print( "\nMedian.\n")
    q2 = df_data.quantile([0.5])
    print(q2)
    print("")
    display(q2)
    # determining the name of the file
    dfq2_excel = 'Q2_Train.xlsx'
    # saving to excel
    q2.to_excel(dfq2_excel)
    print('DataFrame is written to Excel File successfully.')

    print( "\nFirst Quartile.\n")
    q1 = df_data.quantile([0.25])
    print(q1)
    print("")
    display(q1)
    # determining the name of the file
    dfq1_excel = 'Q1_Train.xlsx'
    # saving to excel
    q1.to_excel(dfq1_excel)
    print('DataFrame is written to Excel File successfully.')
    #dfq1 = pd.DataFrame({'Variables': ['x', 'y1', 'y2', 'y3', 'y4'],
                         #'Values': [-10.025, -20.312566, -19.64256, -9.999807, -0.19824]})
    #print(dfq1)
    
    
    print("\nThird Quartile.\n")
    q3 = df_data.quantile([0.75])
    print(q3)
    print("")
    display(q3)
    # determining the name of the file
    dfq3_excel = 'Q3_Train.xlsx'
    # saving to excel
    q3.to_excel(dfq3_excel)
    print('DataFrame is written to Excel File successfully.')
    #dfq3 = pd.DataFrame({'Variables': ['x', 'y1', 'y2', 'y3', 'y4'],
                         #'Values': [9.925, 19.606209, 19.478971, 9.992209, 0.236008]})
    #print(dfq3)
    
    # Convert Q1 to a matrix.
    print("")
    print("Convert Q1 to a matrix.")
    print("")
    matrix_q1 = q1.values
    print(matrix_q1)
    
    # Convert Q3 to a matrix.
    print("")
    print("Convert Q3 to a matrix.")
    print("")
    matrix_q3 = q3.values
    print(matrix_q3)
    
    # Calculate IQR = Q3 - Q1
    print("IQR")
    print("")
    IQR = matrix_q3 - matrix_q1
    print(IQR)
    
    # Convert the IQR matrix to a Pandas DataFrame.
    print("")
#     print("Convert the IQR matrix to an Ideal Pandas DataFrame.")
#     df_IQR_Ideal = pd.DataFrame(IQR, columns=['x', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10',
#                                         'y11', 'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20',
#                                        'y21', 'y22', 'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30',
#                                        'y31', 'y32', 'y33', 'y34', 'y35', 'y36', 'y37', 'y38', 'y39', 'y40',
#                                        'y41', 'y42', 'y43', 'y44', 'y45', 'y46', 'y47', 'y48', 'y49', 'y50'])
    
#     print("Convert the IQR matrix to a Test Pandas DataFrame.")
#     df_IQR_Test = pd.DataFrame(IQR, columns=['x', 'y'])
#     print("Convert the IQR matrix to a Train Pandas DataFrame.")
#     df_IQR_Train = pd.DataFrame(IQR, columns=['x', 'y1', 'y2', 'y3', 'y4'])
#     # Print the DataFrame
#     print("")
#     print(df_IQR_Train)
#     # determining the name of the file
#     df_IQR_excel = 'IQR_Train.xlsx'
#     # saving to excel
#     df_IQR_Train.to_excel(df_IQR_excel)
#     print('DataFrame is written to Excel File successfully.')
    
    print("")
    print("1.5 * IQR")
    print("")
    df_cIQR = 1.5 * IQR
    print(df_cIQR)
    print("")
#     print("Convert the 1.5*IQR matrix to an Ideal Pandas DataFrame.")
#     df_cIQR_Ideal = pd.DataFrame(df_cIQR, columns=['x', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10',
#                                         'y11', 'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20',
#                                        'y21', 'y22', 'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30',
#                                        'y31', 'y32', 'y33', 'y34', 'y35', 'y36', 'y37', 'y38', 'y39', 'y40',
#                                        'y41', 'y42', 'y43', 'y44', 'y45', 'y46', 'y47', 'y48', 'y49', 'y50'])

#     print("Convert the 1.5*IQR matrix to a Test Pandas DataFrame.")
#     df_cIQR_Test = pd.DataFrame(df_cIQR, columns=['x', 'y'])
#     print("Convert the 1.5*IQR matrix to a Train Pandas DataFrame.")
#     df_cIQR_Train = pd.DataFrame(df_cIQR, columns=['x', 'y1', 'y2', 'y3', 'y4'])
    
    # Print the DataFrame.
#     print("")
#     print(df_cIQR_Train)
#     # determining the name of the file
#     df_cIQR_excel = 'cIQR_Train.xlsx'
#     # saving to excel
#     df_cIQR_Train.to_excel(df_cIQR_excel)
#     print('DataFrame is written to Excel File successfully.')    
    
    #Calculate your lower bound.
    print("")                     
    print("Lower bound")
    print("")  
    df_IQR_lower_bound = matrix_q1 - df_cIQR
    print(df_IQR_lower_bound)
    print("")
#     print("Convert the lower bound matrix to an Ideal Pandas DataFrame.")
#     df_IQR_lower_bound_Ideal = pd.DataFrame(df_IQR_lower_bound, columns=['x', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10',
#                                         'y11', 'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20',
#                                        'y21', 'y22', 'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30',
#                                        'y31', 'y32', 'y33', 'y34', 'y35', 'y36', 'y37', 'y38', 'y39', 'y40',
#                                        'y41', 'y42', 'y43', 'y44', 'y45', 'y46', 'y47', 'y48', 'y49', 'y50'])    

#     print("Convert the lower bound matrix to a Test Pandas DataFrame.")
#     df_IQR_lower_bound_Test = pd.DataFrame(df_IQR_lower_bound, columns=['x', 'y'])
#     print("Convert the lower bound matrix to a Train Pandas DataFrame.")
#     df_IQR_lower_bound_Train = pd.DataFrame(df_IQR_lower_bound, columns=['x', 'y1', 'y2', 'y3', 'y4'])

#     # Print the DataFrame.
#     print("")
#     print(df_IQR_lower_bound_Train)
#     # determining the name of the file
#     df_IQR_lower_bound_excel = 'IQR_lower_bound_Train.xlsx'
#     # saving to excel
#     df_IQR_lower_bound_Train.to_excel(df_IQR_lower_bound_excel)
#     print('DataFrame is written to Excel File successfully.') 


    
    #Calculate your upper bound.
    print("")                     
    print("Upper bound")
    print("")  
    df_IQR_upper_bound = matrix_q3 + df_cIQR
    print(df_IQR_upper_bound)
    print("")
#     print("Convert the upper bound matrix to an Ideal Pandas DataFrame.")
#     df_IQR_upper_bound_Ideal = pd.DataFrame(df_IQR_upper_bound, columns=['x', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10',
#                                         'y11', 'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20',
#                                        'y21', 'y22', 'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30',
#                                        'y31', 'y32', 'y33', 'y34', 'y35', 'y36', 'y37', 'y38', 'y39', 'y40',
#                                        'y41', 'y42', 'y43', 'y44', 'y45', 'y46', 'y47', 'y48', 'y49', 'y50'])    

#     print("Convert the upper bound matrix to a Test Pandas DataFrame.")
#     df_IQR_upper_bound_Test = pd.DataFrame(df_IQR_upper_bound, columns=['x', 'y'])
#     print("Convert the upper bound matrix to a Train Pandas DataFrame.")
#     df_IQR_upper_bound_Train = pd.DataFrame(df_IQR_upper_bound, columns=['x', 'y1', 'y2', 'y3', 'y4'])

#     # Print the DataFrame.
#     print("")
#     print(df_IQR_upper_bound_Train)
#     # determining the name of the file
#     df_IQR_upper_bound_excel = 'IQR_upper_bound_Train.xlsx'
#     # saving to excel
#     df_IQR_upper_bound_Train.to_excel(df_IQR_upper_bound_excel)
#     print('DataFrame is written to Excel File successfully.')     
    

    # Remember to change a file_name.
    df_data.plot()
    plt.title('Line plot of Train DataFrame')
    plt.xlabel('Index')
    plt.ylabel('Train values')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    

    # Remember to change columns of the box plot for different file_names.
    fig, at = plt.subplots()
    # Ideal set up of y1-y10.
#     at.boxplot = df_data.boxplot(column =['x', 'y1', 'y2', 'y3', 'y4', 'y5', 'y6', 'y7', 'y8', 'y9', 'y10'])
    # Ideal set up of y11-y20.
#     at.boxplot = df_data.boxplot(column =['y11', 'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20'])
    # Ideal set up of y21-y30.
#     at.boxplot = df_data.boxplot(column =['y21', 'y22', 'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30'])
#     # Ideal set up of y31-y40.
#     at.boxplot = df_data.boxplot(column =['y31', 'y32', 'y33', 'y34', 'y35', 'y36', 'y37', 'y38', 'y39', 'y40'])
#     # Ideal set up of y41-y50.
#     at.boxplot = df_data.boxplot(column =['y41', 'y42', 'y43', 'y44', 'y45', 'y46', 'y47', 'y48', 'y49', 'y50'])
    # Train set up.
#     at.boxplot = df_data.boxplot(column =['x', 'y1', 'y2', 'y3', 'y4'])
    # Test set up.
    at.boxplot = df_data.boxplot(column =['x', 'y'])
    at.set_title('Box plot of columns in Test dataframe without the outliers.')
    at.set_xlabel('Test columns')
    at.set_ylabel('Test values')
    plt.show
    
    # Remember to change columns of the box plot for different file_names.
#     fig, aty = plt.subplots()
#     aty.boxplot = df_data.boxplot(column =['y43', 'y45', 'y46', 'y47'])
#     aty.set_title('Box plot of some columns in Ideal dataframe with outliers.')
#     aty.set_xlabel('Ideal columns')
#     aty.set_ylabel('Ideal values')
#     plt.show

```


```python
#![Jupyter-Python%203-1.png](attachment:Jupyter-Python%203-1.png)
```


```python
#![Python-programming-300x160.png](attachment:Python-programming-300x160.png)
```
