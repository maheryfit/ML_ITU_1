import streamlit as st
from sklearn.model_selection import train_test_split

from my_linear_regression import MyLinearRegression
from service import get_df, pre_treatment_for_training, pre_treatment_for_predict

if __name__ == '__main__':
    maison = get_df()
    maison_2 = get_df("Location-de-maison-Antananarivo-Données-Nomena.csv")
    lr = MyLinearRegression()
    x, y = pre_treatment_for_training(maison)
    lr.fit(x, y)
    x_predict = pre_treatment_for_predict(maison_2, lr)
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
