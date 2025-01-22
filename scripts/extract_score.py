import os

dir = "runlogs"
start = 1737397700

logs = []
for file in os.listdir(dir):
    if int(file.split("_")[0]) < start:
        continue
    logs.append(file)

for file in sorted(logs):
    scores = []
    with open(f"{dir}/{file}") as f:
        for line in f:
            if "test_mean_score" in line:
                scores.append(float(line.split(" ")[-1][:-1]))
    top5 = sorted(scores)[-5:]
    print(f"{sum(top5)/5:.2f}", "_".join(file.split("_")[1:-1]))