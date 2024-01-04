import pandas as pd 
import numpy as np 
import pickle
import time
from sklearn.linear_model import LinearRegression
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import NearMiss 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def Prediction(model):
    y_pred = model.predict(X_test)

    threshold = 0.5
    y_pred = (y_pred > threshold).astype(int)

    st.write('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
    st.write('Precision: {:.3f}'.format(precision_score(y_test, y_pred)))
    st.write('Recall: {:.3f}'.format(recall_score(y_test, y_pred)))
    st.write('F1-score: {:.3f}'.format(f1_score(y_test, y_pred)))
    st.write('ROC-AUC: {:.3f}'.format(roc_auc_score(y_test, y_pred)))

data = st.file_uploader("Выберите файл датасета", type=["csv"])
if data is not None:
    st.header("Датасет")
    df = pd.read_csv(data)
    st.dataframe(df)

    st.write("---")

    feature = st.selectbox("Выберите предсказываемый признак",df.columns)

    st.title("Тип модели обучения")
    model_type = st.selectbox("Выберите тип", [ None, 'Knn', 'Kmeans','Boosting', 'Bagging','Stacking', 'MLP' ])

    button_clicked = st.button("Обработка данных и предсказание")
    if button_clicked:
        st.header("Обработка данных")

        df = df.drop_duplicates()
        
        for i in df.columns[:-1]:
            df[i]=df[i].map(lambda x: np.random.uniform(int(df.min()), int(df.max())) if pd.isna(x) else x)

        scaler = StandardScaler()
        data_scaler = scaler.fit_transform(df.drop(feature, axis=1))

        y = df[feature]
        X = df.drop([feature], axis=1)
        X, _, y, _ = train_test_split(X, y, test_size=0.9, random_state=42)

        nm = NearMiss()
        X, y = nm.fit_resample(X, y.ravel())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        st.success("Обработка завершена")

        st.header("Предсказание")

        if model_type is not None:
            if model_type == "Выберите модель":
                st.write('Для получения предсказания')
            elif model_type == "Knn":
                    with open('models/knn.pkl', 'rb') as file:
                        knn_model = pickle.load(file)
                    Prediction(knn_model)
            elif model_type == "Kmeans":
                    with open('models/kmeans.pkl', 'rb') as file:
                        kmeans_model = pickle.load(file)
                    Prediction(kmeans_model)
            elif model_type == "Boosting":
                    with open('models/boosting.pkl', 'rb') as file:
                        boos_model = pickle.load(file)
                    Prediction(boos_model)
            elif model_type == "Bagging":
                    with open('models/bagging.pkl', 'rb') as file:
                        bagg_model = pickle.load(file)
                    Prediction(bagg_model)
            elif model_type == "Stacking":
                    with open('models/stacking.pkl', 'rb') as file:
                        stac_model = pickle.load(file)
                    Prediction(stac_model)
            elif model_type == "MLP":
                    with open('models/mlp.pkl', 'rb') as file:
                        mlp_model = pickle.load(file)
                    Prediction(mlp_model)
