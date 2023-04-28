import csv
import json

csv_file = 'contrast_set.csv'
json_file = 'contrast_set.json'

# Read CSV file
with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    data = [row for row in csv_reader]

# Write JSON file
with open(json_file, 'w') as file:
    json.dump(data, file, indent=4)
