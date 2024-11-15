**Full name:** Lesedi Kopeledi Matshehla  
    
**Unit testing framework.**  
    
**Description of the Task:** Python-program that uses training data to choose the four ideal
functions which are the best fit out of the fifty provided (C) *. You get (A) 4 training datasets and
(B) one test dataset, as well as (C) datasets for 50 ideal functions. All data respectively consists of
x-y-pairs of values.


**Unit-test code of all 50 ideal functions.**


```python
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
        self.assertEqual(self.df_summary.loc['mean','y1'], -0.002282363249999295, "The mean value of y1 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y2'], 0.045609215055, "The mean value of y2 column"+
                     " was truly computed")             
        self.assertEqual(self.df_summary.loc['mean','y3'], 9.9977176375, "The mean value of y3 column"+
                     " was truly computed")                     
        self.assertEqual(self.df_summary.loc['mean','y4'], 5.045609232, "The mean value of y4 column"+
                     " was truly computed")         
        self.assertEqual(self.df_summary.loc['mean','y5'], -9.9977176375, "The mean value of y5 column"+
                     " was truly computed")         
        self.assertEqual(self.df_summary.loc['mean','y6'], 0.002282363249999295, "The mean value of y6 column"+
                     " was truly computed")         
        self.assertEqual(self.df_summary.loc['mean','y7'], -0.054390777963499996, "The mean value of y7 column"+
                     " was truly computed")         
        self.assertEqual(self.df_summary.loc['mean','y8'], 0.0307260006, "The mean value of y8 column"+
                     " was truly computed")         
        self.assertEqual(self.df_summary.loc['mean','y9'], 0.09121843113, "The mean value of y9 column"+
                     " was truly computed") 
        self.assertEqual(self.df_summary.loc['mean','y10'], -0.36205663077499994, "The mean value of y10 column"+
                     " was truly computed")                     
        self.assertEqual(self.df_summary.loc['mean','y12'], 1.85, "The mean value of y12 column"+
                     " was truly computed")                     
        self.assertEqual(self.df_summary.loc['mean','y13'], -5.099999999999999, "The mean value of y13 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y14'], 0.049999999999999434, "The mean value of y14 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y15'], 3.0249999999999995, "The mean value of y15 column"+
                     " was truly computed")                     
        self.assertEqual(self.df_summary.loc['mean','y16'], 133.335, "The mean value of y16 column"+
                     " was truly computed")         
        self.assertEqual(self.df_summary.loc['mean','y17'], -133.335, "The mean value of y17 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y18'], 266.67, "The mean value of y18 column"+
                     " was truly computed")   
        self.assertEqual(self.df_summary.loc['mean','y19'], 143.335, "The mean value of y19 column"+
                     " was truly computed") 
        self.assertEqual(self.df_summary.loc['mean','y20'], 142.035, "The mean value of y20 column"+
                     " was truly computed")  
        self.assertEqual(self.df_summary.loc['mean','y21'], -20.0, "The mean value of y21 column"+
                     " was truly computed")   
        self.assertEqual(self.df_summary.loc['mean','y22'], 2000.05, "The mean value of y22 column"+
                     " was truly computed") 
        self.assertEqual(self.df_summary.loc['mean','y23'], 20.0, "The mean value of y23 column"+
                     " was truly computed") 
        self.assertEqual(self.df_summary.loc['mean','y24'], -40.0, "The mean value of y24 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y25'], -55.0, "The mean value of y25 column"+
                     " was truly computed")   
        self.assertEqual(self.df_summary.loc['mean','y26'], 787.41, "The mean value of y26 column"+
                     " was truly computed") 
        self.assertEqual(self.df_summary.loc['mean','y27'], 828.61, "The mean value of y27 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y28'], -20.049999999999855, "The mean value of y28 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y29'], 113.33499999999985, "The mean value of y29 column"+
                     " was truly computed") 
        self.assertEqual(self.df_summary.loc['mean','y30'], -281.67, "The mean value of y30 column"+
                     " was truly computed")                     
        self.assertEqual(self.df_summary.loc['mean','y31'], 10.0, "The mean value of y31 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y32'], 2.9810999359828, "The mean value of y32 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y33'], 10.423256193500002, "The mean value of y33 column"+
                     " was truly computed") 
        self.assertEqual(self.df_summary.loc['mean','y34'], 0.5070004845431, "The mean value of y34 column"+
                     " was truly computed")  
        self.assertEqual(self.df_summary.loc['mean','y35'], 8.526512829121202e-16, "The mean value of y35 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y36'], 50.0, "The mean value of y36 column"+
                     " was truly computed") 
        self.assertEqual(self.df_summary.loc['mean','y37'], -10.0, "The mean value of y37 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y38'], -0.04789158018424998, "The mean value of y38 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y39'], 133.33271742251, "The mean value of y39 column"+
                     " was truly computed") 
        self.assertEqual(self.df_summary.loc['mean','y40'], 234.289390960375, "The mean value of y40 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y43'], 1.933862925375, "The mean value of y43 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y44'], -0.01013142119125, "The mean value of y44 column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y45'], 11.933862884999998, "The mean value of y45 column"+
                     " was truly computed")  
        self.assertEqual(self.df_summary.loc['mean','y46'], 4.236448031, "The mean value of y46 column"+
                     " was truly computed")                     
        self.assertEqual(self.df_summary.loc['mean','y47'], -4.236448031, "The mean value of y47 column"+
                     " was truly computed")           
        self.assertEqual(self.df_summary.loc['mean','y49'], 0.029571229945000704, "The mean value of y49 column"+
                     " was truly computed") 
        self.assertEqual(self.df_summary.loc['mean','y50'], 0.040335506957499996, "The mean value of y50 column"+
                     " was truly computed")                     
         
         
        # The chosen four functions with best fit based on train dataset.             
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
```

**Unit-test code of the chosen best fit of four ideal functions based on train dataset.**


```python
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
```

**Test dataset unit-test.**


```python
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
        #self.dataWrangler = DataWrangler("test.csv")
        self.dataset = pd.read_csv("test.csv")
        self.df_data = pd.DataFrame(self.dataset)
        self.assertNotEqual(isinstance(self.df_data, pd.DataFrame), True, "The returned value is of type DataFrame")
        # self.assertEqual(isinstance(self.df_data, pd.DataFrame), True, "The returned value is of type DataFrame")
        
    def test_shape_of_data(self):
        '''
        test the shape_data method that it successfully
        returns the shape of the dataframe constructed
        '''
        self.dataset = pd.read_csv("test.csv")
        self.df_data = pd.DataFrame(self.dataset)
        df_shape = self.df_data.shape
        self.assertEqual(df_shape[0], 100, "The tuple contains at index 0, the value 100,"+
                     " which is our number of rows")
        self.assertEqual(df_shape[1], 2, "The tuple contains at index 1, the value 2,"+
                     " which is our number of columns")
    
    def test_summary_statistics(self):
        '''
        test the summary_statistics method that it successfully
        returns a dataframe bearing the summary statistics of each dataframe column
        '''
        self.dataset = pd.read_csv("test.csv")
        self.df_data = pd.DataFrame(self.dataset)
        self.df_summary = self.df_data.describe()
        self.assertEqual(self.df_summary.loc['mean','x'], 0.299000000000003, "The mean value of x column"+
                     " was truly computed")
        self.assertEqual(self.df_summary.loc['mean','y'], 0.3254828044499999, "The mean value of y column"+
                     " was truly computed")
        
        
if __name__ == "__main__":
    unittest.main()
```

**Train dataset unit-test.**


```python
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
```


```python

```
