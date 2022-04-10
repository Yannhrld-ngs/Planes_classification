
# PLANES IDENTIFICATION
## Table of contents
* [Description of the projet](#description)
* [Technologies](#technologies)
* [Files](#files)
* [Extansion of the model](#extansion)



## Description of the project <a name="description"></a>
The main goal of this project is to classify planes using a picture of a plane. \
We have considered 3 kinds of classe label for the classification :  

- manufacturer
*Here we will classify planes based on the manufacturer.\
More concretely, it means that we will try to find a plane's manufacturer from a simple picture of a plane. \
Example of manufacturer : **Boeing***

- family
*we will classify aircrafts based on the model family from a plane picture.\
A family of planes is a subclass of planes made by a manufactuer. Taking our previous example with Boeing,  
a family can be **Boeing 737***



- variant 
*Here we will classify planes based on model variants. A model variant is a subclass of a family of plane. Still taking our example with Boeing, a model variant can be **Boeing 737-700*** 


For those who are not familiar with planes terminologies, the links below explain and illustrate what a model variant is.
https://simpleflying.com/boeing-737-variants/
https://www.flycovered.com/aircraft/faqs?srch-term=What+do+you+mean+by+model+and+variant
https://en.wikipedia.org/wiki/Category:Lists_of_aircraft_variants


We can visualize our classification using the following url **ADD THE URL**
To classify a picture, go to this url, select a target (manufacturer,variant or family) and then select a model. Afterwards, click on **identify**. To get informations about the probabilities associated to others class labels, click on **other probabilities**


## Technologies <a name="technologies"></a>
To build our classifier, we used various tools. To build the classifier we used an **an artificla neural network** built with **Python(keras and tensorflow)**. To display the classification and deploy the model, we used **streamlit** 



## Files <a name="files"></a>
This project comes with some extra files : 
- ML_model.ipynb
    - This Notebook contains the code used to build our artifical neural networks, the performance of the model and other information. Check it to know more.

- download_data.ipynb
    - This notebook contains the code we used to download the data. 

- streamlit_app.py 
    - This python file contains the code used to generate our streamlit application.

- requirements.txt 
    - This file lists all the modules and packages we used to build our model

- config.yaml
    - This file contains information(directory, values, etc. ) of the required inputs used to build our model in the notebook

- app.yaml
    - This file contains information(directory, values, etc. ) of the required inputs used to build our streamlit app.


## Extansion of the model<a name="extansion"></a>
The model have been built such that we can retrain a new model using a new dataset of planes pictures 
and a new target (whatsoever the target). To build a new model process like this : \
*Step 1:*
- Go to the file **config.yaml**
- Add the path of your dataset just after **DATA_DIR :**.
    - The path must not include previous information
    - Eg : **DATA_DIR : dataset/data** ==> THIS WORKS 
    - Eg : **DATA_DIR : /Users/yannharold/Documents/GitHub/Planes classification/dataset/data** ==> THIS DOESN'T WORK
- Add the target name  by proceeding as before.
    - Eg : **TARGET_NAME : manufacturer**
- Add the name you want to give to the model by proceeding as before.
    - Eg : **MODEL_NAME : epoch_100_model**
- Add the path were you want to save the model by proceeding as before.
    - Eg : **MODEL_DIR : model/results**
- Add the name of your train and test dataset and all other informations required for the algorithm to work (epoch , heigth , width and deepth of the picture ) by proceeding as before.

*Step 2:*
- Go to the file **app.yaml**
- Add the path where you saved the model by proceeding as before. 
    - Eg : **MODEL_DIR : model/results**
- Add information about the heigth , width and deepth of the picture.


More description about the dataset used to build the model  are given in the following link.
https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
