import pandas as pd
import math
import math

target = "loyer_mensuel"
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


def superficie_into_float(df):
    df['superficie'] = df['superficie'].apply(lambda x: x.replace(",", ".") if type(x) == str else x).astype(float)
    return df

def meuble_into_oui_non(df):
    df['meublé'] = df['meublé'].apply(lambda x: "oui" if x == "True" else ("non" if x == "False" else x))
    return df

def loyer_mensuel_fillna(df, predict = False):
    if not predict:
        df['loyer_mensuel'] = df['loyer_mensuel'].fillna(df['loyer_mensuel'].mean())
        return df
    return df

def etat_general_into_bon_mauvais_moyen(df, predict = False):
    if not predict:
        etat_general = [
            ("mauvais", 0, 300000),
            ("moyen", 300000, 800000),
            ("bon", 800000, math.inf)
        ]
        df['état_général'] = df['loyer_mensuel'].apply(
            lambda x: etat_general[0][0] if etat_general[0][1] <= x < etat_general[0][2] else (
                etat_general[1][0] if etat_general[1][1] <= x < etat_general[1][2] else etat_general[2][0]))
        return df
    return df

def douche_wc_separate(df):
    """
    mapping_meuble = {
        "interieur": 5,
        "exterieur": 0
    }
    df['douche_wc'] = df['douche_wc'].replace(mapping_meuble)
    return df
    """
    one_hot_douche_wc = pd.get_dummies(df['douche_wc'])
    df = one_hot_douche_wc.join(df)
    return df.drop('douche_wc',axis = 1)

def superficie_fillna(df):
    quantile_superficie = df['superficie'].quantile([0.25,0.5,0.75])
    df['superficie'] = df['superficie'].fillna(df.apply(lambda x: quantile_superficie[0.25] if x['état_général'] == "mauvais" else (quantile_superficie[0.5] if x['état_général'] == "moyen" else quantile_superficie[0.75]), axis = 1))
    return df

def aberrante_value_superficie(df, predict = False):
    if not predict:
        """
        z_scores = np.abs(stats.zscore(df['superficie']))
        
        # Set a threshold value, say 3
        threshold = 2
        
        # Identify outliers
        outliers = df[z_scores > threshold]
        
        # Filter out the outliers
        return df[z_scores <= threshold]
        """
        quantile_superficie = df['superficie'].quantile([0.25,0.5,0.75])
        Q1_superficie = quantile_superficie[0.25]
        Q3_superficie = quantile_superficie[0.75]
        IQR_super = Q3_superficie - Q1_superficie
        # Define the lower and upper thresholds
        lower_bound_superficie = Q1_superficie - .5 * IQR_super
        upper_bound_superficie = Q3_superficie + .5 * IQR_super
        return df[(df['superficie'] > lower_bound_superficie) & (df['superficie'] < upper_bound_superficie)]
    return df

def aberrante_value_loyer_mensuel(df, predict = False):
    if not predict:
        """
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.float64, np.int64])))

        # Set a threshold value, say 3
        threshold = 3

        # Identify outliers
        outliers = df[z_scores > threshold]
        print(outliers.to_string())
        # Filter out the outliers
        return df[z_scores <= threshold]
        """
        quantile_loyer = df['loyer_mensuel'].quantile([0.25,0.5,0.75])
        Q1_loyer_mensuel = quantile_loyer[0.25]
        Q3_loyer_mensuel = quantile_loyer[0.75]
        IQR_loyer_mensuel = Q3_loyer_mensuel - Q1_loyer_mensuel
        # Define the lower and upper thresholds
        lower_bound_loyer_mensuel = Q1_loyer_mensuel - .5 * IQR_loyer_mensuel
        upper_bound_loyer_mensuel = Q3_loyer_mensuel + .5 * IQR_loyer_mensuel
        return df[(df['loyer_mensuel'] > lower_bound_loyer_mensuel) & (df['loyer_mensuel'] < upper_bound_loyer_mensuel)]
    return df

def etat_general_into_numerical(df):
    mapping_etat_general = {
        "bon": 10,
        "moyen": 5,
        "mauvais": 0
    }
    pd.set_option('future.no_silent_downcasting', True)
    df['état_général'] = df['état_général'].replace(mapping_etat_general)
    return df

def type_d_acces_separate(df):
    mapping_meuble = {
        "sans": 0,
        "moto": 5,
        "voiture": 10,
        "voiture_avec_parking": 15
    }
    df['type_d_acces'] = df['type_d_acces'].replace(mapping_meuble)
    return df
    """
    one_hot_type_acces = pd.get_dummies(df['type_d_acces'])
    df = one_hot_type_acces.join(df)
    return df.drop('type_d_acces',axis = 1)
    """

def meuble_into_numerical(df):
    df['meublé'] = df['meublé'].fillna("non")
    mapping_meuble = {
        "oui": 5,
        "non": 0
    }
    df['meublé'] = df['meublé'].replace(mapping_meuble)
    return df

def quartier_remove(df):
    return df.loc[:, df.columns != 'quartier']

def standardization(df, columns):
    correlation_norm = df.corr()
    correlation_norm = correlation_norm[target].abs().sort_values()
    strong_corr_norm = correlation_norm[(correlation_norm > 0.35)]
    corr_math_norm = df[strong_corr_norm.index].corr()
    features_standardization = corr_math_norm.index
    print(features_standardization)
    columns["colonne"] = features_standardization
    return scaler.fit_transform(df[features_standardization])

def common_pre_treatment(df, predict = False):
    df_pre_trait = superficie_into_float(df)
    df_pre_trait = meuble_into_oui_non(df_pre_trait)
    df_pre_trait = loyer_mensuel_fillna(df_pre_trait, predict)
    df_pre_trait = etat_general_into_bon_mauvais_moyen(df_pre_trait, predict)
    df_pre_trait = douche_wc_separate(df_pre_trait)
    df_pre_trait = superficie_fillna(df_pre_trait)
    df_pre_trait = aberrante_value_superficie(df_pre_trait, predict)
    df_pre_trait = aberrante_value_loyer_mensuel(df_pre_trait, predict)
    df_pre_trait = etat_general_into_numerical(df_pre_trait)
    df_pre_trait = type_d_acces_separate(df_pre_trait)
    df_pre_trait = meuble_into_numerical(df_pre_trait)
    df_pre_trait = quartier_remove(df_pre_trait)
    return df_pre_trait

def pre_treatment_for_predict(df, columns):
    df_pre_trait = common_pre_treatment(df, True)
    for column in columns['colonne']:
        if column not in df_pre_trait.columns:
            df_pre_trait[column] = 0
    return df_pre_trait[columns["colonne"]]

def fill_data_with_zero(df, columns):
    for column in columns:
        if column not in df.columns:
            df[column] = 0
    return df

def pre_treatment_for_training(df, columns):
    df_pre_trait = common_pre_treatment(df)
    y_targ = df_pre_trait[target]
    df_pre_trait = standardization(df_pre_trait, columns)
    return df_pre_trait, y_targ

def get_df(filename: str = "Location de maison Antananarivo  - Données finales - 1.csv"):
    return pd.read_csv(filename)
