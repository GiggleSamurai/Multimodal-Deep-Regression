import json

with open('tiktok_meta_data.json', 'r') as file:
    json_data = json.load(file)

search_id = ["7241650246751669546"]

for id in search_id:
    matching_row = next((row for row in json_data if row.get("id") == id), None)

    if matching_row:
        print("Matching row:")
        print(matching_row)
    else:
        print("No matching row found.")