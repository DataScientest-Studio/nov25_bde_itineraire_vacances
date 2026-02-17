# src/pipelines/run_etl.py
import sys
import time


def run_step(name, module_path):
    print(f"\n==================== {name} ====================")
    start = time.time()
    mod = __import__(module_path, fromlist=["main"])
    if not hasattr(mod, "main"):
        raise RuntimeError(f"{module_path} n'a pas de main()")
    mod.main()
    print(f"✔ {name} terminé en {round(time.time() - start, 2)} sec")


def main():
    steps = [
        ("LOAD TRIPADVISOR (parquet final)", "src.pipelines.01_load_tripadvisor_france"),
        ("LOAD DATATOURISME PRIME (parquet final)", "src.pipelines.02_load_datatourisme_france_prime"),]

    for name, mod in steps:
        run_step(name, mod)

    print("\n✅ RUN ELT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
