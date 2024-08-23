import streamlit as st
import pandas as pd
import numpy as np
import pickle

with open('model_xgb.pkl', 'rb') as file_1:
  model = pickle.load(file_1)

def run():
    with st.form("prediction_form"):
       age = st.number_input('Enter age!',min_value=10,max_value=150)
       work = st.radio('Enter Work Class!',(' State-gov', ' Self-emp-not-inc', ' Private', ' Federal-gov',
       ' Local-gov',' Self-emp-inc', ' Without-pay', ' Never-worked'))
       final = st.number_input('Enter the number of people the census believes the entry represents', max_value=1484705)
       edu = st.radio('Enter Education!',(' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th', ' Some-college',
       ' Assoc-acdm', ' Assoc-voc', ' 7th-8th', ' Doctorate', ' Prof-school',
       ' 5th-6th', ' 10th', ' 1st-4th', ' Preschool', ' 12th'))
       edu_num = st.radio('Enter Education number!',(13,  9,  7, 14,  5, 10, 12, 11,  4, 16, 15,  3,  6,  2,  1,  8))
       marit = st.radio('Enter Marital Status!',(' Never-married', ' Married-civ-spouse', ' Divorced',
       ' Married-spouse-absent', ' Separated', ' Married-AF-spouse',
       ' Widowed'))
       ocu = st.radio('Enter Occupation!',(' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners',
       ' Prof-specialty', ' Other-service', ' Sales', ' Craft-repair',
       ' Transport-moving', ' Farming-fishing', ' Machine-op-inspct',
       ' Tech-support',' Protective-serv', ' Armed-Forces',
       ' Priv-house-serv'))
       rel = st.radio('Enter Relationship!',(' Not-in-family', ' Husband', ' Wife', ' Own-child', ' Unmarried',
       ' Other-relative'))
       race = st.radio('Enter Race!',(' White', ' Black', ' Asian-Pac-Islander', ' Amer-Indian-Eskimo',
       ' Other'))
       sex = st.radio('Enter Sex!',(' Male', ' Female'))
       cap = st.number_input('Enter Capital Gains!')
       capi = st.number_input('Enter Capital Loss!')
       hour = st.number_input('Enter Work Hours Per Week!')
       native = st.radio('Enter Native Country!',(' United-States', ' Cuba', ' Jamaica', ' India', ' Mexico',
       ' South', ' Puerto-Rico', ' Honduras', ' England', ' Canada', ' Germany',
       ' Iran', ' Philippines', ' Italy', ' Poland', ' Columbia', ' Cambodia',
       ' Thailand', ' Ecuador', ' Laos', ' Taiwan', ' Haiti', ' Portugal',
       ' Dominican-Republic', ' El-Salvador', ' France', ' Guatemala',
       ' China', ' Japan', ' Yugoslavia', ' Peru',
       ' Outlying-US(Guam-USVI-etc)', ' Scotland', ' Trinadad&Tobago',
       ' Greece', ' Nicaragua', ' Vietnam', ' Hong', ' Ireland', ' Hungary',
       ' Holand-Netherlands'))

        
       submitted = st.form_submit_button("Submit")


       data_inf = {
            'age': age,
            'workclass': work,
            'fnlwgt': final,
            'education': edu,
            'education-num': edu_num,
            'marital-status': marit,
            'occupation': ocu,
            'relationship': rel,
            'race': race,
            'sex': sex,
            'capital-gain': cap,
            'capital-loss': capi,
            'hours-per-week': hour,
            'native-country': native,
        }
       data_inf = pd.DataFrame([data_inf])
       data_inf
       if submitted:
          result= model.predict(data_inf)
          if result == 0:
             st.write('Salary : <=50K')
          else :
             st.write('Salary : >50K')    

if __name__ == '__main__':
   run()