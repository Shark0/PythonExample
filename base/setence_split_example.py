import re


def split_by_punctuation(sentence):
    # 根據句號、問號、驚嘆號來斷句（保留標點符號）
    sentences = re.split(r'([。！？])', sentence)
    # 將標點與句子重新組合
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        result.append(sentence.strip())
    return result


# 測試文本
text = "今天天氣真好！你吃飯了嗎？我們一起去散步吧。"

# 斷句
lines = split_by_punctuation(text)

# 輸出結果
for line in lines:
    print(line)