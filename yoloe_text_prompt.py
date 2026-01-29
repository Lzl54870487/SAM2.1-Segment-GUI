from ultralytics import YOLO

# 1. 載入預訓練的 YOLOE-26n-seg 模型
model = YOLO("./models/yoloe-26n-seg.pt")
#model = YOLO("./models/yoloe-26n-seg-pf.pt")

# 2. 定義想要偵測的類別（Text Prompts）
# 您可以自由替換為任何想偵測的物體，例如 ["dog", "cat", "car"]
names = ["person"]

# 3. 設置類別與文字嵌入（Text Embeddings）
# 這一點至關重要，用於將文字提示轉換為模型可理解的特徵
model.set_classes(names, model.get_text_pe(names))

# 4. 執行推論
results = model.track("./test_data/雨天無作物.mp4", show=True, save=True, conf=0.6)
