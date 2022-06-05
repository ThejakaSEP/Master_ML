import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


#Display Title and subtitle of the webpage
st.write("""
# Diabetes Detection
Detect if someone has diabetes using machine learning and Python!
""")

#Open and Display an Image
image = Image.open('/Users/thejakamahaulpatha/Downloads/thumb-1920-535242.jpeg')
st.image(image,caption='ML',use_column_width=True)


#Load data to Data Frame
data = pd.read_csv('/Users/thejakamahaulpatha/PycharmProjects/MasterML/diabetes.csv')

#Set a subheader
st.subheader('Data Imformation')

#Show Data
st.dataframe(data)

#Show Statistics
st.write(data.describe())

#Show the data as a Chart
Chart = st.bar_chart(data['Age'])

#Split the data to Independent X and Dependent Y

x=data.iloc[:,0:8].values
y=data.iloc[:,-1].values

#Split the data into 75% Training and 25% Test

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)

#Get the feature input from the user
def get_user_input():
    pregnencies = st.sidebar.slider('Pregnencies',0,17,3)
    Glucose = st.sidebar.slider('Glucose',0,199,117)
    BloodPressure = st.sidebar.slider('Blood_Pressure',0,122,72)
    SkinThickness = st.sidebar.slider('Skin_Thickness',0,99,23)
    Insulin = st.sidebar.slider('Insulin',0.0,846.0,30.0)
    BMI = st.sidebar.slider('BMI',0.0,67.1,32.0)
    DPF = st.sidebar.slider('DPF',0.078,2.42,0.3725)
    Age = st.sidebar.slider('Age',21,100,29)

    #Store a dictionary into a variable
    user_data = {
        'Pregnencies' : pregnencies,
        'Glucose' : Glucose,
        'BloodPressure' : BloodPressure,
        'SkinThickness' : SkinThickness,
        'DPF' : DPF,
        'Insulin' : Insulin,
        'BMI' : BMI,
        'Age' : Age

    }

    #Transform the data into a data frame
    features = pd.DataFrame(user_data,index=[0])
    return features

#Store the user input into the variables

user_input = get_user_input()

#Set a subheader and display user input
st.subheader('User Input :')
st.write(user_input)

#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train,y_train)

#Show the models metrics
st.subheader('Model Test Accuracy Score : ')
st.write(str(accuracy_score(y_test,RandomForestClassifier.predict(x_test))*100)+' %')

#Store the Model prediction in a variable
prediction = RandomForestClassifier.predict(user_input)

#set a subheader and display the classification
st.subheader('Classification : ')
st.write(float(prediction))