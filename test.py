import json

with open('datasets/11422_all_news.json', encoding='utf-8') as json_file:
    output = json.loads(json_file.read())

with open("datasets/first_10_news.json", 'w', encoding='utf8') as out_file:
    json.dump(output[:10], out_file, ensure_ascii=False, indent=4)
