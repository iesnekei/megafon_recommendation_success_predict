#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier, Pool

indexes = ['vas_id', '1', '3', '4', '5', '18', '20', '22', '31', '38', '39', '42',
       '59', '68', '108', '112', '114', '115', '116', '117', '124', '125',
       '130', '134', '136', '138', '144', '145', '146', '165', '167', '192',
       '193', '194', '195', '196', '198', '200', '201', '205', '207', '210',
       '224', '225', '226', '235', '236', '240', '241', '245']

cat_feat = ['vas_id', '31', '192', '194', '195', '196', '198', '200', '201', '205']
# In[4]:


with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# In[7]:


def get_predict(file:str):
    df = pd.read_csv(file)
    try:
        df = df.drop('Unnamed: 0', axis=1)
    except:
        print("Didn't find Unnamed coulumn")
        
    
    info = pd.read_csv('features.csv', sep='\t', header=0)
    
    try:
        info = info.drop('Unnamed: 0', axis=1)
    except:
        print("Didn't find Unnamed coulumn")
    
    df = df.merge(info, on='id', how='left')

    user_id = df.id.tolist()
    buy_time = df.buy_time_x.tolist()
    
    df = df[indexes]
    
    for cat in cat_feat:
        df[cat] = df[cat].apply(lambda x: int(x))
    
    predict = model.predict_proba(df)
    
    predict = np.where(predict[:, 1] > 0.4, 1, 0)
    
    result = pd.DataFrame({'buy_time': buy_time, 'id': user_id, 'vas_id': df.vas_id.tolist(), 'target': predict})
    
    result.to_csv('answers_test.csv', index=False)
    
    return None
    
    


# In[ ]:

if __name__ == "__main__":
    get_predict('data_test.csv')


# In[ ]:




