import pandas as pd

df = pd.read_csv(
    "communes-france-2025.csv",
    usecols=["nom_standard", "code_postal", "latitude_centre", "longitude_centre", "reg_nom", "dep_nom"],
    dtype={
        "nom_standard": str,
        "code_postal": str,
        "latitude_centre": float,
        "longitude_centre": float,
        "reg_nom": str,
        'dep_nom': str
    },
)

regions = ['Auvergne-Rhône-Alpes', 'Île-de-France', 'Bretagne']

df= df.loc[df['reg_nom'].isin(regions)]

df = df.rename(
    columns={
        "nom_standard": "locality_name",
        "code_postal": "postal_code",
        "latitude_centre": "center_latitude",
        "longitude_centre": "center_longitude",
        "reg_nom": "reg_name",
        "dep_nom": "dep_name"
    }
)

df["locality"] = df[["locality_name", "postal_code", 'reg_name']].apply(
    lambda row: f"{row['locality_name']} ({row['postal_code']}), {row['reg_name']}", axis=1
)

df.to_csv("localities.csv")
