# fix_labels.py
input_file = "/scratch/penghao/datasets/Date-Real/labels.txt"
output_file = "/scratch/penghao/datasets/Date-Real/labels_fixed.txt"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        parts = line.strip().split()
        if len(parts) >= 2:
            img = parts[0]
            label = " ".join(parts[1:])  # rejoin as full label
            fout.write(f"{img}\t{label}\n")
