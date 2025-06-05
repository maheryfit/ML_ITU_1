import streamlit as st
from sklearn.model_selection import train_test_split

from my_linear_regression import MyLinearRegression
from service import get_df, pre_treatment_for_training, pre_treatment_for_predict
import pandas as pd
COLUMNS = ['quartier', 'superficie', 'nombre_chambres', 'douche_wc', 'type_d_acces', 'meublé', 'état_général']
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    maison = get_df("Location de maison Antananarivo  - Données finales - 1.csv")
    maison_2 = get_df("Location-de-maison-Antananarivo-Données-Nomena.csv")
    columns = {}
    lr = MyLinearRegression()
    x, y = pre_treatment_for_training(maison, columns)
    lr.fit(x, y)
    #x_predict = pre_treatment_for_predict(maison_2, lr)
    data = [
        ['Ivandry', 70.0, 2, 'exterieur', 'sans', 'oui', 'mauvais'],
        ['Ivandry', 300.0, 5, 'interieur', 'sans', 'oui', 'bon'],
        ['Ivandry', 8.0, 3, 'interieur', 'voiture_avec_parking', 'non', 'bon']
    ]
    x_predict = pd.DataFrame(data, columns=COLUMNS)
    x_predict = pre_treatment_for_predict(x_predict, columns)
    print(lr.predict(x_predict))
    """x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    lr = MyLinearRegression()
    lr.fit(x_train, y_train)
    lr.predict(x_test)
    root_mse = lr.get_rmse(x_test, y_test)
    score = lr.get_r2_score(x_test, y_test)
    print(root_mse)
    print(score)
    """
