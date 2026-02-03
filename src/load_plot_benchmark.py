from benchmark.benchmark_io import load_benchmark
from benchmark.visualize_benchmark import (
    plot_order_stability,
    plot_size_comparison,
    plot_stability,
)

results = load_benchmark()

plot_size_comparison(results["size_comparison"], save_path="graphs/size_comparison.png")

plot_stability(results["stability_ga"], "ga", save_path="graphs/stability_ga.png")

plot_order_stability(
    results["stability_ga"], "ga", save_path="graphs/order_stability_ga.png"
)

plot_stability(results["stability_nn2o"], "nn2o", save_path="graphs/stability_nn2o.png")

plot_order_stability(
    results["stability_nn2o"], "nn2o", save_path="graphs/order_stability_nn2o.png"
)
