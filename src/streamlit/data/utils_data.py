import pandas as pd

df = pd.read_csv(
    "data/communes-france-2025.csv",
    usecols=["nom_standard", "code_postal", "latitude_centre", "longitude_centre"],
    dtype={
        "nom_standard": str,
        "code_postal": str,
        "latitude_centre": float,
        "longitude_centre": float,
    },
)


df = df.rename(
    columns={
        "nom_standard": "locality_name",
        "code_postal": "postal_code",
        "latitude_centre": "center_latitude",
        "longitude_centre": "center_longitude",
    }
)

df["locality"] = df[["locality_name", "postal_code"]].apply(
    lambda row: f"{row['locality_name']} ({row['postal_code']})", axis=1
)
df.to_csv("localities.csv")
