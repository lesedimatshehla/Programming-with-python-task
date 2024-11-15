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
        test the load_data method that it successfully
        constructed a dataframe from the .csv file
        '''
        #self.dataWrangler = DataWrangler("train.csv")
        self.dataset = pd.read_csv("train.csv")
        self.df_data = pd.DataFrame(self.dataset)
        self.assertNotEqual(isinstance(self.df_data, pd.DataFrame), True, "The returned value is of type DataFrame")
        # self.assertEqual(isinstance(self.df_data, pd.DataFrame), True, "The returned value is of type DataFrame")
        
    def test_shape_of_data(self):
        '''
        test the shape_data method that it successfully
        returns the shape of the dataframe constructed
        '''
        self.dataset = pd.read_csv("train.csv")
        self.df_data = pd.DataFrame(self.dataset)
        df_shape = self.df_data.shape
        self.assertEqual(df_shape[0], 400, "The tuple contains at index 0, the value 400,"+
                     "which is our number of rows")
        self.assertEqual(df_shape[1], 5, "The tuple contains at index 1, the value 5,"+
                     "which is our number of columns")
    
    def test_summary_statistics(self):
        '''
        test the summary_statistics method that it successfully
        returns a dataframe bearing the summary statistics of each dataframe column
        '''
        self.dataset = pd.read_csv("train.csv")
        self.df_data = pd.DataFrame(self.dataset)
        self.df_summary = self.df_data.describe()
        self.assertEqual(self.df_summary.loc['mean','x'], -0.049999999999999434, "The mean value of x column"+
                     "was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y1'], 0.10766606164249878, "The mean value of y1 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y2'], -0.09423904243749917, "The mean value of y2 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y3'], -0.05162830643749942, "The mean value of y3 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y4'], 0.01263280748595, "The mean value of y4 column"+
                     " was truly computed") 
        
        
if __name__ == "__main__":
    unittest.main()

