import re
import statistics

text = open("/scratch/penghao/OpenOCR/output_inference/inference_results.txt").read()

# Extract all floating‑point numbers that follow "time_cost"
time_costs = [float(x) for x in re.findall(r'time_cost":\s*([0-9.]+)', text)]

print("Count:", len(time_costs))
print("Average time_cost:", statistics.mean(time_costs))
dates = re.findall(r'"date":\s*"[0-9\-]+"', text)
print("Number of dates:", len(dates))