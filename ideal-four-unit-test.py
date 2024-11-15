#!/usr/bin/env python
# coding: utf-8

# **Full name:** Lesedi Kopeledi Matshehla  
#     
# **Unit testing framework.**  
#     
# **Description of the Task:** Python-program that uses training data to choose the four ideal
# functions which are the best fit out of the fifty provided (C) *. You get (A) 4 training datasets and
# (B) one test dataset, as well as (C) datasets for 50 ideal functions. All data respectively consists of
# x-y-pairs of values.
# 

# In[ ]:


import unittest
import pandas as pd

class TestDataWrangler(unittest.TestCase):
    '''
    Test the DataWrangler Class
    '''
    def test_load_data(self):
        '''
        four best fit ideal functions
        test the load_data method that it successfully
        constructed a dataframe from the .csv file
        '''
        #self.dataWrangler = DataWrangler("ideal.csv")
        self.dataset = pd.read_csv("ideal.csv")
        self.df_data = pd.DataFrame(self.dataset)
        self.assertNotEqual(isinstance(self.df_data, pd.DataFrame), True, "The returned value is of type DataFrame")
        # self.assertEqual(isinstance(self.df_data, pd.DataFrame), True, "The returned value is of type DataFrame")
        
    def test_shape_of_data(self):
        '''
        four best fit ideal functions 
        test the shape_data method that it successfully
        returns the shape of the dataframe constructed
        '''
        self.dataset = pd.read_csv("ideal.csv")
        self.df_data = pd.DataFrame(self.dataset)
        df_shape = self.df_data.shape
        self.assertEqual(df_shape[0], 400, "The tuple contains at index 0, the value 400,"+
                     " which is our number of rows")
        self.assertEqual(df_shape[1], 51, "The tuple contains at index 1, the value 51,"+
                     " which is our number of columns")
    
    def test_summary_statistics(self):
        '''
        four best fit ideal functions
        test the summary_statistics method that it successfully
        returns a dataframe bearing the summary statistics of each dataframe column
        '''
        self.dataset = pd.read_csv("ideal.csv")
        self.df_data = pd.DataFrame(self.dataset)
        self.df_summary = self.df_data.describe()
        self.assertEqual(self.df_summary.loc['mean','x'], -0.049999999999999434, "The mean value of x column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y42'], 0.12280453926250061, "The mean value of y42 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y41'], -0.10114118499999904, "The mean value of y41 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y11'], -0.049999999999999434, "The mean value of y11 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y48'], -0.0004656956999996426, "The mean value of y48 column"+
                     " was truly computed")
        
        
if __name__ == "__main__":
    unittest.main()

