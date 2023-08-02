#!/usr/bin/env python
# coding: utf-8

# In[1]:
import re
import pandas as pd

def assay_name_divider(df,feature,regex,new_feature_name):
    '''meant to extract relevant information from a column and create a new column with information from regex
    Parameters
    -----------
    df = dataframe 
    feature = column # to search the regex in
    regex = string literal in format: r'(?<=)strip_design' to search with
    new_feature_name = string literal which names the new column
    
    Returns an updated dataframe'''
    pattern = re.compile(regex)
    
    finder_list = []
    for var in df.iloc[:,feature]:
        finder = pattern.findall(var)
        finder_list.append(finder[0])
        
    df.insert(feature+1, new_feature_name, finder_list)
    return df


# In[ ]:




