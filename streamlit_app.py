#%% LOADING LIBRARIES
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import yaml 
from PIL import Image
import pandas as pd
from matplotlib.pyplot import style
import os
#%% FUNCTIONS
def load_image(path):
    """Load an image as numpy array
    """
    return plt.imread(path)
    

def predict_image(path, model,IMAGE_WIDTH, IMAGE_HEIGHT):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction
    
    Returns
    -------
    Predicted class
    """
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    print(images.shape)
    prediction_vector = model.predict(images)
    predicted_classes = np.argmax(prediction_vector, axis=1)
    return predicted_classes[0] , prediction_vector


def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)

#%% STREAMLIT APP CONSTRUCTION...

#Reading the configuration yaml file
with open('app.yaml') as f :
    app_config = yaml.load(f, Loader=yaml.FullLoader)

path = app_config['MODEL_DIR']

#GETING MODELS
#Get all files present in a directory and subdirectory and take only those with extension.h5
filelist = []
for root, dirs, files in os.walk(path):
    for file in files :
        file_path = os.path.join(root,file)
        if os.path.splitext(file_path)[1] == '.h5' :
            filelist.append(file_path )

#Get subdirectory ==> get target names
subdirectory = [ os.path.basename( os.path.dirname(file) )  for file in  filelist ]


st.title("PLANES IDENTIFICATION")

uploaded_file = st.file_uploader("Please, load a picture of a plane") #, accept_multiple_files=True)#

with st.sidebar:
    st.title("model selection : ")
    
    #Step 1 : target selection
    add_radio = st.radio(
        "Select first a target",
        tuple(set(subdirectory)) #set to avoid to duplicate the name of a target
    )
    
    #Step 2 : model selection for a corresponding target
    
    #ADD A CONDITION SUCH THAT WE CHOSE A MODEL ONLY AFTER TARGET SELECTION
    
    models_path = [ file for file in filelist if add_radio == os.path.basename( os.path.dirname(file) ) ]
    models_name = [ os.path.basename(file) for file in filelist if add_radio == os.path.basename( os.path.dirname(file) ) ]
    
    add_radio_2 = st.radio(
        "Then select a model",
        tuple(set(models_name))
    )
    
    our_model = [file for file in models_path if os.path.basename(file) == add_radio_2 ][0]
    
    
if uploaded_file:
    loaded_image = load_image(uploaded_file)
    st.image(loaded_image)

#Loading the model 
model = load_model(our_model) 
#model.summary()

#Loading correspondinh target of a model 
#Get corresponding labels of the classification
target_path = os.path.dirname(our_model)+'/'+os.path.basename(our_model).replace('.h5','_target.txt')
file2 = open(target_path) #path to access to the target names
file2 = file2.read().split('\n')
file2 = file2[:len(file2)-1]


predict_btn = st.button("Identify", disabled=(uploaded_file is None))
if predict_btn:
    prediction , proba_vector = predict_image(uploaded_file , model, app_config['IMAGE_WIDTH'] , app_config["IMAGE_HEIGTH"] )
    proba = proba_vector[0].max()
    st.write(f" Prediction : { file2[prediction] }")
    st.write(f"Probability associated with the prediction :{proba}")
    
    
info_btn = st.button("Other probabilities", disabled=(uploaded_file is None) )
if info_btn:
    prediction , proba_vector = predict_image(uploaded_file , model, app_config['IMAGE_WIDTH'] , app_config["IMAGE_HEIGTH"] )
    proba = proba_vector[0].max()
    fig, ax = plt.subplots( figsize = (30 , 10) )
    style.use('seaborn-poster') #sets the size of the charts
    style.use('ggplot')
    x = file2
    y = proba_vector[0]
    ax.bar(x ,y , color = 'darkgreen')
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(fig)


#%% STREAMLIT APP DEPLOYMENT
# cd /Users/yannharold/Documents/GitHub/Planes_classification 
# streamlit run streamlit_app.py
