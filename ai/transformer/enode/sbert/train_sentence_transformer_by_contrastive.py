from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses


def get_model():
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return model


def train_model(model):
    train_examples = [
        InputExample(texts=["商店休假日期、商店決定於2025年1月26日至2025年2月9日期間進行休假", "更改休息日日設定"],
                     label=1.0),
        InputExample(texts=["商店休假日期、商店決定於2024年1月26日至2024年2月9日期間進行休假", "更改休息日日設定"],
                     label=1.0),
        InputExample(texts=["我們希望把2間商店H6608001及H6608002於2025年所有公眾假期設定為休息日", "更改休息日日設定"],
                     label=1.0),
        InputExample(texts=["我們希望把2間商店H6608001及H6608002於2024年所有公眾假期設定為休息日", "更改休息日日設定"],
                     label=1.0),
        InputExample(texts=["我們希望把2間商店H6608002及H6608003於2025年所有公眾假期設定為休息日", "更改休息日日設定"],
                     label=1.0),
        InputExample(texts=[
            "申請2025年公眾假期休息日、因為早前錯過了申請2025年公眾假期休息日的 email、現想申請2025年所有的公眾假期休息",
            "更改休息日日設定"], label=1.0),
        InputExample(texts=[
            "申請2024年公眾假期休息日、因為早前錯過了申請2025年公眾假期休息日的 email、現想申請2025年所有的公眾假期休息更改休息日日設定",
            "更改休息日日設定"], label=1.0),
        InputExample(texts=["休息日、申請年初三（星期五）、年初四（星期六）2日為休息日", "更改休息日日設定"], label=1.0),
        InputExample(texts=["休息日、申請年初三、年初四2日為休息日", "更改休息日日設定"], label=1.0),
        InputExample(texts=["農歷新年假期休息、我司需要申請2025之休息日", "更改休息日日設定"], label=1.0),
        InputExample(texts=["農歷新年假期休息、我司需要申請2023之休息日", "更改休息日日設定"], label=1.0),
        InputExample(texts=["申請1月28日至2月3日改為非工作日(放假)", "更改休息日日設定"], label=1.0),
        InputExample(texts=["申請2月28日至2月23日改為非工作日(放假)", "更改休息日日設定"], label=1.0),
        InputExample(texts=["下列日期維休息日", "更改休息日日設定"], label=1.0),
        InputExample(texts=["下列日期維休息日不出貨", "更改休息日日設定"], label=1.0),
        InputExample(texts=["我想吃冰淇淋", "更改休息日日設定"], label=0),
        InputExample(texts=["農歷新年我想打籃球", "更改休息日日設定"], label=0),
        InputExample(texts=["2間商店H6608002及H6608003要賣內褲", "更改休息日日設定"], label=0),
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=15)

    train_loss = losses.ContrastiveLoss(model)
    # warmup_steps = int(len(train_dataloader) * 3 * 0.1)
    # print("warmup_steps: ", warmup_steps)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=10,
    )
    model.save("./model_support_001-1")


def get_trained_model():
    model = SentenceTransformer('./model_support_001-1')
    return model


def cluster(model):
    labels = [
        "收不到訂單通知/ 不能列印運單",
        "發現漏貨/缺貨",
        "運輸/ GOGOVAN交貨查詢",
        "[商戶派送] 出貨前 協助商戶聯絡客人",
        "出貨後問題 查詢3R/ 退件運費",
        "查詢訂單被CS Cancel 之原因",
        "包裝物料訂單查詢",
        "設定/修飾店舖頁面",
        "產品上架問題",
        "於MMS找不到合適的品牌/顏色/尺碼/店名/ 產品類別/ 產地",
        "更改商戶店舖的網頁路徑 (URL)",
        "更改運單上的商戶名稱",
        "新增海外地區/ 國家標籤",
        "快速抗原測試產品",
        "食品和飲料產品（包括酒類）",
        "保健產品",
        "藥劑及毒藥產品",
        "中醫藥",
        "除害劑產品",
        "電子產品",
        "消費品或玩具產品",
        "成人用品",
        "其他產品類品",
        "EXCHANGE系統問題",
        "MMS系統問題",
        "收不到電郵通知/ 更改MMS 聯絡資料",
        "商品評論",
        "店舖評分",
        "更改領取日期設定 (即商戶入倉日子)",
        "更改休息日日設定",
        "合約期查詢/ 開分店",
        "特惠佣金",
        "續約查詢",
        "新增BB code/ 修改BB頁面中的內容",
        "換購/贈品/大手折扣申請或更改",
        "長期紅價/ 有限期折扣優惠/ 任選組合優惠",
        "參與Marketing活動",
        "使用HKTVmall logo製作宣傳品",
        "廣告/有關廣告投放系統查詢",
        "貨款/週期報表狀況",
        "訂單狀況差異",
        "佣金率差異",
        "其它調整/罰款差異",
        "其他貨款查詢",
        "收款銀行資料查詢",
        "罰款查詢",
        "方案及收費",
        "入倉預訂和商戶營運查詢",
        "倉存及派送查詢",
        "系統查詢",
        "3PL其他查詢",
    ]
    inputs = [
        "商店休假日期、商店決定於2025年1月26日至2025年2月9日期間進行休假"
    ]

    label_embeddings = model.encode(labels)
    input_embeddings = model.encode(inputs)
    similarity = model.similarity(input_embeddings, label_embeddings)
    print(similarity)

    similarity_list = similarity.flatten().tolist()
    paired = [{l: s} for l, s in zip(labels, similarity_list)]
    paired_sorted = sorted(paired, key=lambda d: list(d.values())[0], reverse=True)
    print(paired_sorted)
    print("\n")


def main():
    model = get_model()
    print("before training")
    cluster(model)
    train_model(model)
    trained_model = get_trained_model()
    print("after training")
    cluster(trained_model)


if __name__ == "__main__":
    main()
