from etl.extract import extract_all
from etl.merge import merge_dataframes
from etl.transform import transform
from etl.save import save_parquet, save_tables_csv
from etl.embedding.h3_indexer import add_h3_columns
from etl.scoring.density import add_density
from etl.scoring.proximity import add_proximity
from etl.scoring.diversity import add_diversity
from etl.scoring.category_weight import add_category_weight
from etl.scoring.opening_hours import add_opening_hours_score
from etl.scoring.popularity import add_popularity
from etl.scoring.final_score import add_final_score
from etl.utils.bounding_box import BoundingBoxResolver
from etl.embedding.embeddings import build_text_embedding_column, add_embeddings


import time


def main():
    resolver = BoundingBoxResolver()
    start_total = time.perf_counter()

    df_list = extract_all()
    df = merge_dataframes(df_list)
    df = transform(df)

    print(df.head())
    print(f"Total rows: {df.shape[0]}")

    # add h3 column for location filtering
    df = add_h3_columns(df, lat_col="latitude", lon_col="longitude")

    # scoring
    df = (
        df
        .pipe(add_density, level="commune")
        .pipe(add_diversity, level="commune")
        .pipe(add_popularity)
        .pipe(add_proximity, resolver, level="commune")
        .pipe(add_category_weight)
        .pipe(add_opening_hours_score)
        .pipe(add_final_score)
    )

    # Construire la colonne texte riche
    #df = build_text_embedding_column(df)

    # Ajouter les embeddings
    #df = add_embeddings(df)

    # Save final dataset in parquet and csv
    save_parquet(df)
    save_tables_csv(df)

    end_total = time.perf_counter()
    print(f"\n=== Temps total du process : {end_total - start_total:.2f} sec ===")

if __name__ == "__main__":
    main()