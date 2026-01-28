from sklearn.cluster import KMeans

def cluster_poi(poi_df, num_days) :
    
    X = poi_df[['longitude', 'latitude']]
   
    kmeans = KMeans(n_clusters= num_days,
                     random_state= 43, 
                     n_init= 'auto')
    poi_df['day_cluster'] = kmeans.fit_predict(X)

    return poi_df
