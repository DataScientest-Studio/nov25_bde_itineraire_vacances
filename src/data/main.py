from etl.extract import extract_all
from etl.merge import merge_dataframes
from etl.transform import transform
from etl.save import save_parquet
import time


def main():
    start_total = time.perf_counter()

    df_list = extract_all()
    df = merge_dataframes(df_list)
    df = transform(df)

    print(df.head())
    print(f"Total rows: {df.shape[0]}")

    end_total = time.perf_counter()
    print(f"\n=== Temps total du process : {end_total - start_total:.2f} sec ===")

    df = save_parquet(df)


    end_total = time.perf_counter()
    print(f"\n=== Temps total du process : {end_total - start_total:.2f} sec ===")

if __name__ == "__main__":
    main()