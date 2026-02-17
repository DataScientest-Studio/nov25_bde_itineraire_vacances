# ======================================================================================================================
# DF MASTER (construit à partir des données déjà normalisées) ---> docker/ui/services/providers.py
# ======================================================================================================================

from infra.imports import pd

def build_master_df(df_dt, df_ab, df_ta):
    """
    Construit le DataFrame master utilisé pour :
    - Découverte
    - Recommandation
    - Options sidebar
    Devient la source unique de vérité côté UI.
    """
    df = pd.concat([df_dt, df_ab, df_ta], ignore_index=True)
    return df
