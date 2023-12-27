from ppo import Evaluation
import os

spec_06 = "/Pramana/RL4Real/POSET-RL/data/SPEC2006/"
spec_17 = "/Pramana/RL4Real/POSET-RL/data/spec-17/"

benchmark_suites = {"spec_06": spec_06, "spec_17": spec_17}
benchmarks = {"spec_06": [], "spec_17": []}

for suite in benchmark_suites:
  for filename in os.listdir(benchmark_suites[suite]):
    if filename.endswith(".ll") and filename.startswith(""):
      file_path = benchmark_suites[suite] + filename
      file_size = os.path.getsize(file_path)
#      if file_size < 30000000: # check if file size is less than 30MB
      benchmarks[suite].append({"filename": filename, "path": file_path, "time": [], "size": file_size})
benchmarks.pop("spec_17")
import pandas as pd

for suite in benchmarks:
  benchmarks[suite] = sorted(benchmarks[suite], key=lambda x: x['size'])

import time, random

n_iters = 2

def run(flow_name, flow_func):
  print("Flow: ", flow_name)
  for k, v in benchmarks.items():
    print("Benchmark Suite: ", k)
    csv_file_path = flow_name + "-" + k + ".csv"
    try:
      df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
      df = pd.DataFrame(columns=['Benchmark'] + ['Iteration ' + str(i+1) for i in range(n_iters)] + ['Average Time'])
      print(v)
    for benchmark in v:
      try:
        if benchmark['filename'] not in df['Benchmark'].values:
          benchmark["time"] = []
          print("Benchmark: ", benchmark)
          row = {"Benchmark": benchmark["filename"]}
          for i in range(n_iters):
            start = time.time()
            # Evaluation.eval(benchmark["path"])
            flow_func(benchmark["path"])
            # time.sleep(random.uniform(0, 2))
            end = time.time()
            benchmark["time"].append(end - start)
            row["Iteration " + str(i+1)] = benchmark["time"][i]
            print("Iteration ", i+1, " Time: ", benchmark["time"][i])
          row["Average Time"] = sum(benchmark["time"])/len(benchmark["time"])
          df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
      except Exception as e:
  #      df.to_csv(csv_file_path, index=False)
        print("Exception: ",e)
        continue
 #     exit()
    df.to_csv(csv_file_path, index=False)

# for compilergym
#run("compilergym", Evaluation.eval)

# for bridge
BUILD_DIR= "/home/cs20btech11024/repos/ml-llvm-project/build_release"
ML_CONFIG_PATH = "/home/cs20btech11024/repos/ML-Register-Allocation/build_release/config"
# --ml-config-path={ML_CONFIG_PATH}
run("bridge", lambda file: os.system(f'{BUILD_DIR}/bin/opt  --codesizeopt-rl {file} > /dev/null'))

      
