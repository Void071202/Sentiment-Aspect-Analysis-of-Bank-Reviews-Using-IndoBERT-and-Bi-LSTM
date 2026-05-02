import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import scipy.stats as st
import math

class Sampling:
    """
    Class to get the sampling methods Adjusted Cross Validation (ACV), Time-based sampling (TBS), Adjusted Window 
    Size v1. (AWS1.0), Adjusted Windows Size v2.0 (AWS2.0)
    
    Initial Input
        fold    number of sampling fold; integer > 1; default is fold=10
        alpha   confidence level to get normal ppf; 0-1; default is alpha=0.95
        width   error in unit; integer >=1; default is width=1
        
    Function
        ACV
        Input   
            self
            y   response variable
        Output
            zip(train,test)
            
        
        cochran
        Input
            self
            X   dataset to get original length
            y   dataset to get the variance
        Output
            new_train_size
        
        AWS
        Input
            self
            y   response variable
            v   version; 0,1,2,3 are for ACV, TBS, AWS1.0, AWS2.0 respectively; default is v=3
        Output
            zip(train,test)
    """
    def __init__(self,fold=10,alpha=0.95,width=1):
        self.k = fold
        self.a = alpha
        self.w = width
    
    def ACV(self,y):
        if not isinstance(y,(pd.Series,np.ndarray)):
            raise ValueError("1st argument have to be in array or series")
        k = KFold(n_splits=self.k)
        train_index = [] 
        test_index = []
        
        for train,test in k.split(y):
            if len(train[train<test[0]])!=0:
                train_index.append(train[train<test[0]])
                test_index.append(test)
                
        return zip(train_index,test_index)
    
    def cochran(self,X,y):
        N = len(X)
        v = np.var(y)
        
        Z = st.norm.ppf(self.a)   
        n0  = (4*v*(Z**2))/(self.w**2)
    
        return math.ceil(n0/(1+(n0-1)/(N)))
        
    def AWS(self,y,v=3):
        if not isinstance(y,(pd.Series,np.ndarray)):
            raise ValueError("1st argument have to be in array or series")
        elif v not in [0,1,2,3]:
            raise ValueError('2nd argument have to integer in [0-3]')
        train_index = []
        test_index = []

        for train,test in self.ACV(y):
            if v==0: # adjusted cross validation
                n=len(train)
            elif v==1: # time-based sampling
                n=self.cochran(y,y)
            elif v ==2: # adjusted window size v1 
                n=self.cochran(y,y[train])
            elif v==3: # adjusted window size v2
                n = self.cochran(y[train],y[train])
                
            n = len(train) if n>len(train) else n
            train_index.append(train[-n:])
            test_index.append(test)

        return zip(train_index,test_index)