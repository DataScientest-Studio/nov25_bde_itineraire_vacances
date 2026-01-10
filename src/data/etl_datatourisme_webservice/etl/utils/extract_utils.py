import pandas as pd

# définition d'une fonction pour transformer une liste d'un seul élément en un élement :
def simple_list_extract(x) :
    if (isinstance(x,list)) and (len(x) == 1) :
       return x[0]
    else :
        return x

def vectorized_simple_list_extract(df) :
    return df.apply(lambda col: col.apply(simple_list_extract), axis = 0)

def poi_structure_extract(poi_df, data_struct):
    """ 
    cette fonctione permet de répartir les valeurs d'une clonne de poi_df, qui est sous forme de dictionnaire/liste de données 'data_struct', sur plusieurs colonnes
    la valeur d'une colonne doit avoir un des formats suivants :
    - liste de dictionnaires
    - dictionnaire
    - NaN
        
    """

    df_raw = poi_df[['dc:identifier', data_struct]]
        
    #création de 4 masks pour diviser le dataframe en 4 sections : 

    # sans thème (valeur null) :
    mask_null = df_raw[data_struct].isna()

    # avec theme sous forme de dictionnaire :
    mask_dict = df_raw[data_struct].apply(lambda x : (isinstance(x,dict)))

    # avec une structure hastheme sous forme d'une liste d'un seul élément 
    mask_simple_list = df_raw[data_struct].apply(lambda x : (isinstance(x,list)) and (len(x)==1) )

    # avec une structure hastheme sous forme d'une liste d'un seul élément 
    mask_multiple_list = df_raw[data_struct].apply(lambda x : (isinstance(x,list)) and (len(x) > 1) )

    # répartition des éléments de la liste sur plusieurs lignes, une ligne par élément :
    if mask_multiple_list.sum() > 0 :
        simple_list_df = df_raw[mask_multiple_list].explode(data_struct)
    else :
        simple_list_df = pd.DataFrame()
        
    # concaténation avec les poi ayant une structure 'hasTheme' simple :
    simple_list_df = pd.concat([simple_list_df,
                                df_raw[mask_simple_list].map(simple_list_extract),
                                df_raw[mask_dict] ], axis = 0, ignore_index= True)

    if len(simple_list_df) > 0 :

        # répartition du dictionnaire 'hasTheme' sur plusieurs colonnes, une colonne par clé :
        columns_df = pd.json_normalize(data = simple_list_df[data_struct])

        # ajout des identifiants des POI
        columns_df.index = simple_list_df.index
        df = pd.concat([simple_list_df[['dc:identifier']], columns_df], axis = 1)

        if mask_null.sum() > 0 :
            # ajout des POI sans structure de donnée
            df = pd.concat([df, df_raw[mask_null]], axis = 0, ignore_index= True).map(simple_list_extract)

    else :
        df = simple_list_df

    return df