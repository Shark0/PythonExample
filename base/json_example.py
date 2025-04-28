import json

numbers = [1, 2, 3, 4, 5, 6]

filename = "file/json_example.json"
with open(filename, "w") as file:
    json.dump(numbers, file)

with open(filename) as file:  # 以讀取模式開啟檔案(若沒有第二個參數都是預設成讀取模式)
    json_value = json.load(file)  # 用json.load()載入放在number.json檔裡面的資料，然後將它存到numbers變數中

print(json_value)
