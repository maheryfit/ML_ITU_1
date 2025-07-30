import streamlit as st
import folium

from service import write_csv_file_for_pollution_securisation
from tp_1 import pre_traitement_data_from_form, get_current_latitude_and_longitude

from streamlit_folium import st_folium

def input_pollution():
    return st.number_input(
        "Pollution", value=None, placeholder = "Entrer l'échelle de pollution..."
    )

def input_securisation():
    return st.number_input(
        "Sécurisation", value=None, placeholder = "Entrer l'échelle de sécurisation..."
    )

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


def input_all(geojson_file):
    dict_retour = {
        "quartier": input_quartiers_maps(geojson_file),
        "pollution": input_pollution(),
        "securisation": input_securisation(),
    }
    return dict_retour

if __name__ == '__main__':
    st.write("# Entrer les informations de quartier")
    with st.form("my_form"):
        columns = input_all("TP_1/geoBoundaries-MDG-ADM4.geojson")
        submitted = st.form_submit_button("Entrer les informations sur les quartiers")
        if submitted:
            data = pre_traitement_data_from_form(columns)
            quartier = data[0]
            pollution = data[1]
            securisation = data[2]
            print(quartier, pollution, securisation)
            write_csv_file_for_pollution_securisation(quartier, float(pollution), float(securisation))
            st.write(f"#### Quartier {quartier} enregistré")
