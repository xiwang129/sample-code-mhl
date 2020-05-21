
# coding: utf-8

# In[9]:


import logging, requests, os, io, glob, time, torch 
from space_bandits import load_model
import sagemaker
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder


# In[10]:


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# In[13]:


CSV_CONTENT_TYPE = 'application/csv'


# In[14]:


# loads the model
def model_fn(model_dir):
    logger.info('model_fn')
    path = Path(model_dir)
    
    with open(os.path.join(model_dir,'model_10k.pkl'), 'rb') as f:
        model = load_model(f)
    
    return model


# In[18]:


# deserialize the invoke request body into an obeject that can be perform prediction on
def input_fn(request_body, content_type = CSV_CONTENT_TYPE):

    logger.info('Deserializing the input data.')
    if content_type == CSV_CONTENT_TYPE:
          return io.BytesIO(request_body)
      
    else: 
        raise ValueError('Requested unsupported content type')


# In[ ]:


# perform prediction on the deserialized object with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    
    predict_class,predict_idx,predict_values = model.predict(input_object)
    
    print("--- Inference time: %s seconds ---" % (time.time() - start_time))
    print(f'Predicted class is {str(predict_class)}')
    print(f'Predict confidence score is {predict_values[predict_idx.item()].item()}')
    
    
    return dict(class = str(predict_class),
        confidence = predict_values[predict_idx.item()].item())


# In[20]:


# serialize the prediction result into the desired response content type
def output_fn(prediction, accept=CSV_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == CSV_CONTENT_TYPE: return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))    

