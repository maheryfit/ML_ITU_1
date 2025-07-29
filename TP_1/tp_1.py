import geocoder
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from my_linear_regression import MyLinearRegression
from service import get_df, pre_treatment_for_training, pre_treatment_for_predict

def get_current_latitude_and_longitude():
    g = geocoder.ip('me')#this function is used to find the current information using our IP Add
    if g.latlng is not None: #g.latlng tells if the coordinates are found or not
        latitude, longitude = g.latlng
        return latitude, longitude
    else:
        return None


COLUMNS = ['quartier', 'superficie', 'nombre_chambres', 'douche_wc', 'type_d_acces', 'meublé', 'état_général']

def get_quartiers(df):
    return df['quartier'].unique()[1:]

def input_quartiers_st(df):
    quartiers = get_quartiers(df)
    return st.selectbox("Quartier", tuple(quartiers))

def input_quartiers_maps(geojson_file: str):
    gps = get_current_latitude_and_longitude()
    #print(gps)
    if gps is not None:
        latitude, longitude = gps
        current_map = folium.Map(location=[latitude, longitude],
                   zoom_start=10, control_scale=True)
        popup = folium.GeoJsonPopup(
            fields=["shapeName"],
            aliases=["Area Name:"]
        )
        folium.GeoJson(geojson_file, name="Madagascar", popup=popup).add_to(current_map)
        return st_folium(current_map, width=800, height=300)
    return None


def input_superficie():
    return st.number_input(
        "Superficie", value=None, placeholder = "Entrer la superficie..."
    )

def input_douche_wc():
    return st.radio(
        "Emplacement douche et WC", ["***interieur***", "***exterieur***"]
    )

def input_type_acces():
    return st.radio(
        "Type d'accès", ["***sans***", "***moto***", "***voiture***", "***voiture_avec_parking***"]
    )

def input_etat_general():
    return st.radio(
        "Type d'accès", ["***bon***", "***moyen***", "***mauvais***"]
    )

def input_nombre_chambres():
    return st.number_input(
        "Nombre de chambres", value=None, placeholder = "Entrer le nombre de chambres..."
    )

def input_meuble():
    return st.checkbox(
        "Meublé"
    )

def input_all(geojson_file):
    #input_quartiers_maps(geojson_file)
    dict_retour = {
        "quartier": input_quartiers_maps(geojson_file),
        "superficie": input_superficie(),
        "nombre_chambres": input_nombre_chambres(),
        "douche_wc": input_douche_wc(),
        "type_acces": input_type_acces(),
        "meublé": input_meuble(),
        "etat_general": input_etat_general()
    }
    return dict_retour

def predict(maison_data, dataform):
    columns_data = {}
    x, y = pre_treatment_for_training(maison_data, columns_data)
    linear_regression = MyLinearRegression()
    linear_regression.fit(x, y)
    x_predict = pd.DataFrame([dataform], columns=COLUMNS)
    x_predict = pre_treatment_for_predict(x_predict, columns_data)
    # Display graph weight variables
    st.pyplot(columns_data['plt'].gcf())
    return linear_regression.predict(x_predict)

def get_area_name(data_0):
    try:
        return data_0["last_active_drawing"]["properties"]["shapeName"]
    except Exception as e:
        return None

def pre_traitement_data_from_form(columns_data):
    infos = list(columns_data.values())
    infos = [(str(i).replace("*", "") if type(i) == str else i) for i in infos]
    infos.insert(1, get_area_name(infos[0]))
    del infos[0]
    return infos

if __name__ == '__main__':
    maison = get_df("TP_1/Location de maison Antananarivo  - Données finales - 1.csv")
    st.write("# Prédit le prix de ta maison")
    with st.form("my_form"):
        columns = input_all("TP_1/geoBoundaries-MDG-ADM4.geojson")
        submitted = st.form_submit_button("Prédire la location")
        if submitted:
            data = pre_traitement_data_from_form(columns)
            prediction = predict(maison, data)
            st.write(f"#### Prix prédit: {'{:,}'.format(int(prediction[0]))} Ariary")

def function():
    pass
    """
    columns = {}
    x, y = pre_treatment_for_training(maison, columns)
    lr = MyLinearRegression()
    lr.fit(x, y)
    #x_predict = pre_treatment_for_predict(maison_2, lr)
    data = [
        ['Ivandry', 10.0, 2, 'exterieur', 'sans', 'non', 'mauvais'],
        ['Ivandry', 45.0, 5, 'interieur', 'sans', 'oui', 'bon'],
        ['Ivandry', 89.0, 3, 'interieur', 'voiture_avec_parking', 'non', 'bon']
    ]
    x_predict = pd.DataFrame(data, columns=COLUMNS)
    x_predict = pre_treatment_for_predict(x_predict, columns)
    print(lr.predict(x_predict))
    """
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
    lr = MyLinearRegression()
    lr.fit(x_train, y_train)
    lr.predict(x_test)
    root_mse = lr.get_rmse(x_test, y_test)
    score = lr.get_r2_score(x_test, y_test)
    print(root_mse)
    print(score)
    """
