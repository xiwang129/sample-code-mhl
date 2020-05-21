
# coding: utf-8

# In[24]:


import requests
import pandas as pd
import json
from six import BytesIO
import numpy as np


# In[25]:


df = pd.read_csv('shuttle.txt')


# In[26]:


contexts = df.to_numpy()


# In[27]:


context_json = json.dumps(contexts.tolist())


# In[28]:


d = {'data':context_json}


# In[29]:


api_endpoint = 'https://qw6r43xed7.execute-api.us-east-1.amazonaws.com/test'


# In[30]:


r = requests.post(api_endpoint, json=d)


# In[31]:


r.text

