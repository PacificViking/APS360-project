import json
with open("cards.json", 'r') as file:
    parsed = json.load(file)

print(len(parsed))
#print(json.dumps(parsed, indent=2))
