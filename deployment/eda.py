import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title=' Census Salary EDA & Prediction',
    layout='wide',
    initial_sidebar_state='expanded'
)
def run():
    st.title('Census Salary Prediction')
    st.subheader('EDA to analyze Census Salary Data')
    st.image('data:image/jpeg;base64,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',
             caption='Census Salary')
    st.write('This page was created by Evan Juanto')
    st.markdown('---')


    df = pd.read_csv('salary.csv')
    st.dataframe(df)
    st.write('### Distribution ')
    pilihan = st.selectbox('Pick features:',('age','hours-per-week'))
    fig = plt.figure(figsize=(10,6))
    sns.histplot(df[pilihan],bins=30, kde=True)
    st.pyplot(fig)

    st.write('### Distribution of Education')
    fig = plt.figure(figsize=(10,6))
    sns.countplot(y='education', data=df, order=df['education'].value_counts().index)
    st.pyplot(fig)

    st.write('### Distribution of Marital Status')
    fig = plt.figure(figsize=(10,6))
    df['marital-status'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
    st.pyplot(fig)

    st.write('### Average Hours Per Week based on Marital Status')
    fig = plt.figure(figsize=(10,6))
    sns.pointplot(x='marital-status', y='hours-per-week', data=df)
    st.pyplot(fig)

    st.write('### Relationship of Age and Hours Per Week')
    fig = sns.jointplot(x='age', y='hours-per-week', data=df, kind='scatter')
    plt.suptitle('Hubungan Age dan Hours Per Week', y=1.02)
    st.pyplot(fig)

    st.write('### Age Distribution Based on Salary and Sex')
    g = sns.FacetGrid(df, col='salary', row='sex', margin_titles=True, height=4, aspect=1.5)
    g.map(sns.histplot, 'age', bins=20)
    g.fig.suptitle('Age Distribution Based on Salary and Sex', y=1.03)
    st.pyplot(g)


if __name__ == '__main__':
    run()
