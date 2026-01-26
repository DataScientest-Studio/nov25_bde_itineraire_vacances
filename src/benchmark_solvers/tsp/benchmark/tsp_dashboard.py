import pandas as pd

runner = TSPBenchmarkRunner(distance_matrices, start=0, runs=5)
results = runner.run()

df = pd.DataFrame(results)
df = df.sort_values("rating", ascending=False)
print(df)