with open("file/input.txt") as file_object:
    content = file_object.read()
    print(content.rstrip())

with open("file/input.txt") as file_object:
    for line in file_object:
        print(line)

filename = "file/output.txt"

with open(filename, "w") as file_object:  # 以寫入模式開啟number.txt檔
    file_object.write("Shark")  # 寫入第一行
    file_object.write("Lin")  # 寫入第二行

with open(filename, "a") as file_object:  # 以附加模式開啟number.txt檔
    file_object.write("0918477996")
