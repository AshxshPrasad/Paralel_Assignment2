import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "results" / "benchmark_results.csv"
PLOT_DIR = BASE_DIR / "plots"

PLOT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Read benchmark results
# --------------------------------------------------
df = pd.read_csv(CSV_FILE)

# Exclude 1000-element correctness test
# It is not part of the main performance benchmark.
df = df[df["input_size"] >= 100000].copy()

# Convert input size to millions for easier reading
df["input_size_m"] = df["input_size"] / 1_000_000

# --------------------------------------------------
# Plot 1: CPU vs CUDA Total Execution Time
# --------------------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(
    df["input_size_m"],
    df["cpu_time_ms"],
    marker="o",
    linewidth=2,
    label="CPU"
)

plt.plot(
    df["input_size_m"],
    df["cuda_total_time_ms"],
    marker="o",
    linewidth=2,
    label="CUDA Total"
)

plt.xlabel("Input Size (Millions of Elements)")
plt.ylabel("Execution Time (ms)")
plt.title("CPU vs CUDA Total Execution Time")
plt.xticks(df["input_size_m"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "execution_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# Plot 2: CUDA Kernel Execution Time
# --------------------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(
    df["input_size_m"],
    df["cuda_kernel_time_ms"],
    marker="o",
    linewidth=2,
    label="CUDA Kernel"
)

plt.xlabel("Input Size (Millions of Elements)")
plt.ylabel("Kernel Execution Time (ms)")
plt.title("CUDA Kernel Execution Time vs Input Size")
plt.xticks(df["input_size_m"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "kernel_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# Plot 3: Speedup
# --------------------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(
    df["input_size_m"],
    df["kernel_speedup"],
    marker="o",
    linewidth=2,
    label="Kernel Speedup"
)

plt.plot(
    df["input_size_m"],
    df["end_to_end_speedup"],
    marker="o",
    linewidth=2,
    label="End-to-End Speedup"
)

plt.xlabel("Input Size (Millions of Elements)")
plt.ylabel("Speedup (×)")
plt.title("CUDA Speedup over CPU")
plt.xticks(df["input_size_m"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "speedup.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Visualization complete!")
print(f"Graphs saved to: {PLOT_DIR}")