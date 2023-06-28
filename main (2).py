import numpy as np 
import pickle
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns



pickle_dtree=open("crop_pred_dtree.pkl","rb")
pickle_ran=open("crop_pred_rf.pkl","rb")
pickle_knn=open("crop_pred_knn.pkl","rb")
pickle_svm=open("crop_pred_svc.pkl","rb")
pickle_logistic=open("crop_pred_logistic.pkl","rb")
#pickle_gb=open("crop_pred_gb.pkl","rb")

classifier=pickle.load(pickle_dtree)
classifier1=pickle.load(pickle_ran)
classifier2=pickle.load(pickle_knn)
classifier3=pickle.load(pickle_svm)
classifier4=pickle.load(pickle_logistic)
#classifier5=pickle.load(pickle_gb)





def main():
    st.title("Crop Prediction") 
    html_temp = """
    <div style="background-color:#50C878; padding:10px;">
    <h2 style="color:white; text-align:center;">Predict Your Crop</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    activities=['Decision Tree Classification','Random Forest Classification','K-Nearest Neighbor(KNN)','Support Vector Machine(SVM)','Logistic Regression', 'Gradient Boosting']
    option=st.sidebar.selectbox('which model would you like to use ?',activities)
    st.subheader(option)
    temperature=st.text_input('Input your Temparature here:','Range=[9,55]')
    humidity=st.text_input('Input your Humidity here:','Range=[11,99]')
    ph=st.text_input('Input your Ph here:','Range=[4,9]')
    N=st.text_input('Input your Nitrogen here:','Range=[9,55]')
    P=st.text_input('Input your Phosphorus here:','')
    K=st.text_input('Input your Potassium here:','')
    Rainfall=st.text_input('Input your rainfall here:','')
    
    inputs=[[temperature,humidity,ph,N,P,K,Rainfall]]

    result=""


    if st.button('Predict'):
        if option == 'Random Forest Classification':
            result=classifier1.predict(inputs)[0]
            st.success('Crop Prediction Result : {}'.format(result))
        elif option == 'K-Nearest Neighbor(KNN)':
            result=classifier2.predict(inputs)[0]
            st.success('Crop Prediction Result : {}'.format(result))
        elif option == 'Support Vector Machine(SVM)':
            result=classifier3.predict(inputs)[0]
            st.success('Crop Prediction Result : {}'.format(result))
        elif option == 'Logistic Regression':
            result=classifier4.predict(inputs)[0]
            st.success('Crop Prediction Result : {}'.format(result))
        elif option == 'Gradient Boosting)':
            result=classifier5.predict(inputs)[0]
            st.success('Crop Prediction Result : {}'.format(result))
        else:
            result=classifier.predict(inputs)[0] 
            st.success('Crop Prediction Result : {}'.format(result))



        # result=predict_note_authentication(temperature,humidity,ph)
        # st.success('the output is {}'.format(result))

    if st.button('about'):
        st.text("ISE244 Project by Govind Chennu")
        st.text("Email: saisrigovind.chennu@sjsu.edu")

@st.cache
def load_data(nrows):
    data=pd.read_csv('Crop_recommendation.csv',nrows=nrows)
    return data


data_list=load_data(1000)

st.subheader('CROP DATA')
st.write(data_list)

st.subheader('Label Index')
st.bar_chart(data_list['label'])

df=pd.DataFrame(data_list[:200],columns=['temperature','humidity','ph'])

st.subheader('Variance of Environmental factors')
for col in df:
  fig=plt.figure(figsize = (16, 9))
  sns.boxplot(x = 'label', y = col, data = data_list, palette = 'rocket')
  plt.xlabel('labels', fontsize = 12)
  plt.ylabel(col, fontsize = 12)
  plt.xticks(rotation=90)
  plt.title(f'{col} vs Crop', fontweight='bold')
  st.pyplot(fig)
  





st.subheader('Line Chart of Humidity, pH and Temperature')
st.line_chart(df)

st.subheader('Area chart of pH and Rainfall')
chart_data=pd.DataFrame(data_list[:40],columns=['ph','rainfall'])
st.area_chart(chart_data)




if __name__ == '__main__':
    main()
