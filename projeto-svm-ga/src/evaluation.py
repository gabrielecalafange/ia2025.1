import json
import matplotlib.pyplot as plt

# Load coverage results
with open("coverage.json") as f:
    data = json.load(f)

total_statements = data["totals"]["num_statements"]
covered = data["totals"]["covered_lines"]
missing = total_statements - covered

# Data for the pie chart
labels = ["Covered Lines", "Missing Lines"]
sizes = [covered, missing]

plt.figure(figsize=(6, 6))
plt.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Overall Test Coverage")
plt.tight_layout()
plt.savefig("coverage.png")
