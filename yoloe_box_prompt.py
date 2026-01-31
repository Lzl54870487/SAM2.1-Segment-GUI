import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# 1. 初始化模型與讀取影片首幀
model = YOLO("./models/yoloe-26n-seg.pt")
video_path = "./test_data/WIN_20250911_18_14_48_Pro.mp4"
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
if not success:
    print("無法讀取影片"); exit()

# 2. 拖拽選取邏輯
roi_start, drawing = None, False
clicked_bboxes = []
temp_display = frame.copy()

def mouse_callback(event, x, y, flags, param):
    global roi_start, drawing, temp_display
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_img = frame.copy()
        cv2.rectangle(temp_img, roi_start, (x, y), (0, 255, 0), 2)
        cv2.imshow("Select ROI", temp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = roi_start
        clicked_bboxes.append([min(x1, x), min(y1, y), max(x1, x), max(y1, y)])
        cv2.rectangle(frame, roi_start, (x, y), (255, 0, 0), 2) # 藍色固定框
        cv2.imshow("Select ROI", frame)

# 3. 互動選取 (可調大小視窗)
cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select ROI", mouse_callback)
cv2.imshow("Select ROI", frame)

print("提示：請拖拽滑鼠選取目標，完成後按 'ENTER' 開始推論，按 'ESC' 退出。")
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 13: break # ENTER
    if key == 27: cv2.destroyAllWindows(); exit()

cv2.destroyAllWindows()

# 4. 執行視覺提示推論並過濾結果
if clicked_bboxes:
    # 根據官方規範封裝 Prompt [Model Prediction](https://docs.ultralytics.com/reference/models/yolo/model/#ultralytics.models.yolo.model.YOLOE.predict)
    visual_prompts = {
        "bboxes": np.array(clicked_bboxes),
        "cls": np.array([0] * len(clicked_bboxes))
    }
    
    cv2.namedWindow("Top 1 Object", cv2.WINDOW_NORMAL) # 推論視窗亦可縮放 ✅

    results = model.predict(
        source=video_path,
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        device="cuda",
        stream=True
    )

    for r in results:
        # 僅過濾置信度最高的物件
        if len(r.boxes) > 0:
            top1_idx = r.boxes.conf.argmax()
            r = r[int(top1_idx)] 
        
        # 繪製並顯示結果 [Results.plot](https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Results.plot)
        annotated_frame = r.plot()
        cv2.imshow("Top 1 Object", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
