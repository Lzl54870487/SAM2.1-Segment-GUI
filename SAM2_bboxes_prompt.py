import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from ultralytics.models.sam import SAM2VideoPredictor
import json
import os

class SAM2TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM2 Video Tracker")

        # 初始化變量
        self.prompts = []
        self.roi_start = None
        self.drawing = False
        self.current_rect = None
        self.save_video = False  # 是否儲存視頻的標誌
        self.save_masks_only = False  # 是否儲存僅有Mask的影片的標誌
        self.classes = ["Object"]  # 類別列表，默認為"Object"
        self.current_class_index = 0  # 當前選擇的類別索引
        self.color_map = {}  # 類別顏色映射 (存儲(R, G, B, Alpha)元組)
        self.alpha_map = {}  # 類別透明度映射
        self.generate_color_map()  # 生成初始顏色映射

        # 設定配置文件路徑
        self.config_path = "./sam2_config.json"

        # 選擇影片檔案
        self.video_path = filedialog.askopenfilename(
            title="選擇影片檔案",
            initialdir="./test_data/",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v"),
                ("All files", "*.*")
            ]
        )

        if not self.video_path:
            print("未選擇視頻檔案")
            return

        # 加載視頻和第一幀
        self.cap = cv2.VideoCapture(self.video_path)
        success, self.frame_orig = self.cap.read()

        if not success:
            print("無法讀取視頻檔案")
            return

        self.orig_h, self.orig_w = self.frame_orig.shape[:2]

        # 創建SAM2VideoPredictor
        overrides = dict(
            conf=0.25,
            device='cuda',
            task="segment",
            mode="predict",
            imgsz=1024,
            model="./models/sam2.1_t.pt"
        )
        self.predictor = SAM2VideoPredictor(overrides=overrides)

        # 保存基本配置，以便稍後根據需要創建不同配置的predictor
        self.base_overrides = overrides

        # 設置GUI
        self.setup_gui()

        # 自動加載配置
        self.load_config()

        # 在加載配置後，初始化當前類別的顏色和透明度值到UI控件
        if self.classes:
            # 設置第一個類別為當前選中類別
            self.class_var.set(self.classes[0])
            # 更新UI控件以顯示當前類別的顏色和透明度值
            self.on_class_selected()

    def load_config(self):
        """加載配置檔案"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 加載類別設置
                if 'classes' in config:
                    self.classes = config['classes']

                # 加載顏色映射
                if 'color_map' in config:
                    # 將字串鍵轉換為元組值
                    self.color_map = {}
                    for class_name, color_list in config['color_map'].items():
                        if isinstance(color_list, list) and len(color_list) == 3:
                            self.color_map[class_name] = tuple(color_list)

                # 加載透明度映射
                if 'alpha_map' in config:
                    self.alpha_map = config['alpha_map']

                print(f"配置已加載: {self.config_path}")

            except Exception as e:
                print(f"加載配置檔案時出錯: {e}")
                # 如果加載失敗，使用預設設置
                self.classes = ["Object"]
                self.generate_color_map()
        else:
            # 如果配置檔案不存在，初始化預設設置
            self.classes = ["Object"]
            self.generate_color_map()

    def setup_gui(self):
        # 主容器
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 上半部分：圖像顯示區域
        self.canvas = tk.Canvas(main_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # 綁定鼠標事件
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # 綁定窗口大小改變事件
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # 中間部分：中央執行按鈕
        center_button_frame = ttk.Frame(main_frame)
        center_button_frame.pack(fill=tk.X, pady=(0, 10))

        # 配置按鈕樣式
        style = ttk.Style()
        style.configure("Large.TButton", font=("TkDefaultFont", 12, "bold"))

        # 完成選擇按鈕（加大並置中）
        self.finish_btn = ttk.Button(center_button_frame, text="完成選擇並開始追蹤", command=self.start_tracking, style="Large.TButton")
        self.finish_btn.pack(side=tk.TOP, pady=5)

        # 下半部分：控制面板
        control_panel = ttk.Frame(main_frame)
        control_panel.pack(fill=tk.X, pady=(0, 5))

        # 左側控制區域
        left_control_frame = ttk.Frame(control_panel)
        left_control_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 按鈕框架
        button_frame = ttk.Frame(left_control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 5))

        # 重新選擇檔案按鈕
        self.reselect_btn = ttk.Button(button_frame, text="重新選擇檔案", command=self.reselect_video)
        self.reselect_btn.pack(side=tk.LEFT, padx=(0, 10))

        # 重置按鈕
        self.reset_btn = ttk.Button(button_frame, text="重置選擇", command=self.reset_selections)
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 10))

        # 儲存影片開關按鈕
        self.save_video_btn = ttk.Button(button_frame, text="儲存影片: 否", command=self.toggle_save_video)
        self.save_video_btn.pack(side=tk.LEFT)

        # 儲存僅有Mask影片開關按鈕
        self.save_masks_only_btn = ttk.Button(button_frame, text="儲存Mask影片: 否", command=self.toggle_save_masks_only)
        self.save_masks_only_btn.pack(side=tk.LEFT, padx=(0, 10))

        # 類別控制框架
        class_control_frame = ttk.Frame(left_control_frame)
        class_control_frame.pack(fill=tk.X, pady=(5, 0))

        # 類別數量控制
        ttk.Label(class_control_frame, text="類別數量:").pack(side=tk.LEFT, padx=(0, 5))
        self.class_count_var = tk.StringVar(value=str(len(self.classes)))
        self.class_count_label = ttk.Label(class_control_frame, textvariable=self.class_count_var)
        self.class_count_label.pack(side=tk.LEFT, padx=(0, 10))

        self.add_class_btn = ttk.Button(class_control_frame, text="+", width=3, command=self.add_class)
        self.add_class_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.remove_class_btn = ttk.Button(class_control_frame, text="-", width=3, command=self.remove_class)
        self.remove_class_btn.pack(side=tk.LEFT, padx=(0, 10))

        # 類別選擇下拉選單
        ttk.Label(class_control_frame, text="當前類別:").pack(side=tk.LEFT, padx=(0, 5))
        self.class_var = tk.StringVar()
        self.class_dropdown = ttk.Combobox(class_control_frame, textvariable=self.class_var, state="readonly")
        self.class_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        self.update_class_dropdown()

        # 綁定下拉選單選擇事件
        self.class_dropdown.bind("<<ComboboxSelected>>", self.on_class_selected)

        # 類別命名按鈕
        self.rename_class_btn = ttk.Button(class_control_frame, text="重命名類別", command=self.rename_class)
        self.rename_class_btn.pack(side=tk.LEFT)

        # 右側控制區域 - RGB顏色選擇
        right_control_frame = ttk.Frame(control_panel)
        right_control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # RGB顏色選擇框架
        rgb_control_frame = ttk.Frame(right_control_frame)
        rgb_control_frame.pack(fill=tk.Y, pady=(0, 0), padx=(10, 0))

        # RGB顏色選擇標籤
        ttk.Label(rgb_control_frame, text="類別顏色設置 (RGB):").pack(side=tk.TOP, padx=(0, 5), pady=(0, 5))

        # R/G/B值輸入框架
        rgb_input_frame = ttk.Frame(rgb_control_frame)
        rgb_input_frame.pack(side=tk.TOP, fill=tk.X)

        # R值輸入
        ttk.Label(rgb_input_frame, text="R:").pack(side=tk.LEFT, padx=(5, 0))
        self.r_var = tk.StringVar(value="255")
        self.r_entry = ttk.Entry(rgb_input_frame, textvariable=self.r_var, width=5)
        self.r_entry.pack(side=tk.LEFT, padx=(0, 5))

        # G值輸入
        ttk.Label(rgb_input_frame, text="G:").pack(side=tk.LEFT)
        self.g_var = tk.StringVar(value="0")
        self.g_entry = ttk.Entry(rgb_input_frame, textvariable=self.g_var, width=5)
        self.g_entry.pack(side=tk.LEFT, padx=(0, 5))

        # B值輸入
        ttk.Label(rgb_input_frame, text="B:").pack(side=tk.LEFT)
        self.b_var = tk.StringVar(value="0")
        self.b_entry = ttk.Entry(rgb_input_frame, textvariable=self.b_var, width=5)
        self.b_entry.pack(side=tk.LEFT, padx=(0, 5))

        # 透明度輸入框架
        alpha_input_frame = ttk.Frame(rgb_control_frame)
        alpha_input_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

        # 透明度標籤
        ttk.Label(alpha_input_frame, text="Alpha:").pack(side=tk.LEFT, padx=(5, 0))
        self.alpha_var = tk.StringVar(value="0.5")
        self.alpha_entry = ttk.Entry(alpha_input_frame, textvariable=self.alpha_var, width=5)
        self.alpha_entry.pack(side=tk.LEFT, padx=(0, 5))

        # 透明度範圍提示
        ttk.Label(alpha_input_frame, text="(0-1.0)").pack(side=tk.LEFT)

        # 設置顏色按鈕
        self.set_color_btn = ttk.Button(rgb_control_frame, text="設置顏色", command=self.set_class_color)
        self.set_color_btn.pack(side=tk.TOP, padx=(0, 0), pady=(5, 0))

        # 顯示初始圖像
        self.display_image(self.frame_orig)

    def on_canvas_resize(self, event):
        # 當canvas大小改變時重新顯示圖像
        self.display_image(self.frame_orig)

    def display_image(self, image):
        # 轉換BGR到RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 調整圖像大小以適應canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600

        # 計算縮放比例以保持縱橫比，確保圖像完整顯示在canvas中
        h, w = image_rgb.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 調整圖像大小
        resized_image = cv2.resize(image_rgb, (new_w, new_h))

        # 轉換為PIL Image
        pil_image = Image.fromarray(resized_image)
        self.photo = ImageTk.PhotoImage(pil_image)

        # 在canvas上顯示圖像，居中顯示
        self.canvas.delete("all")  # 清除之前的所有元素
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)

        # 保存縮放比例和偏移量以便後續座標轉換
        self.scale_x = w / new_w
        self.scale_y = h / new_h
        # 計算實際圖像在canvas中的位置（考慮居中）
        self.offset_x = (canvas_width - new_w) // 2
        self.offset_y = (canvas_height - new_h) // 2

        # 重新繪製已有的矩形框
        self.redraw_existing_boxes()

    def redraw_existing_boxes(self):
        # 重新繪製已選擇的框
        for prompt in self.prompts:
            box = prompt['bbox']
            class_name = prompt['class']
            x1, y1, x2, y2 = box
            # 轉換座標到canvas空間
            x1_canvas = x1 / self.scale_x + self.offset_x
            y1_canvas = y1 / self.scale_y + self.offset_y
            x2_canvas = x2 / self.scale_x + self.offset_x
            y2_canvas = y2 / self.scale_y + self.offset_y

            # 获取该类别的颜色
            color = self.color_map.get(class_name, (255, 0, 0))  # 默认红色 (RGB)
            # 将RGB元组转换为十六进制颜色字符串用于tkinter
            if isinstance(color, tuple) and len(color) == 3:
                hex_color = '#%02x%02x%02x' % color
            else:
                hex_color = '#FF0000'  # 默认红色

            self.canvas.create_rectangle(x1_canvas, y1_canvas, x2_canvas, y2_canvas,
                                        outline=hex_color, width=2)
            # 显示类别标签
            self.canvas.create_text(x1_canvas, y1_canvas - 10, text=class_name,
                                   fill=hex_color, anchor='sw', font=('Arial', 10, 'bold'))

    def on_mouse_down(self, event):
        # 轉換canvas座標到原始圖像座標
        x_img = int((event.x - self.offset_x) * self.scale_x)
        y_img = int((event.y - self.offset_y) * self.scale_y)

        # 檢查座標是否在有效範圍內
        if 0 <= x_img < self.orig_w and 0 <= y_img < self.orig_h:
            self.drawing = True
            self.roi_start = (x_img, y_img)

    def on_mouse_drag(self, event):
        if self.drawing and self.roi_start:
            # 清除之前的臨時矩形
            self.canvas.delete("temp_rect")

            # 轉換canvas座標到原始圖像座標
            x_img = int((event.x - self.offset_x) * self.scale_x)
            y_img = int((event.y - self.offset_y) * self.scale_y)

            # 檢查座標是否在有效範圍內
            if 0 <= x_img < self.orig_w and 0 <= y_img < self.orig_h:
                # 轉換回canvas座標進行顯示
                x1_canvas = self.roi_start[0] / self.scale_x + self.offset_x
                y1_canvas = self.roi_start[1] / self.scale_y + self.offset_y
                x2_canvas = x_img / self.scale_x + self.offset_x
                y2_canvas = y_img / self.scale_y + self.offset_y

                # 繪製臨時矩形
                self.canvas.create_rectangle(x1_canvas, y1_canvas, x2_canvas, y2_canvas,
                                            outline='green', width=2, tags="temp_rect")

                # 更新十字準線
                self.update_crosshair(event)

    def on_mouse_move(self, event):
        # 更新十字準線
        self.update_crosshair(event)

    def update_crosshair(self, event):
        # 清除之前的十字準線
        self.canvas.delete("crosshair")

        # 獲取canvas尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # 繪製十字準線
        self.canvas.create_line(0, event.y, canvas_width, event.y, fill='yellow', width=1, tags="crosshair")
        self.canvas.create_line(event.x, 0, event.x, canvas_height, fill='yellow', width=1, tags="crosshair")

    def on_mouse_up(self, event):
        if self.drawing and self.roi_start:
            # 轉換canvas座標到原始圖像座標
            x_img = int((event.x - self.offset_x) * self.scale_x)
            y_img = int((event.y - self.offset_y) * self.scale_y)

            # 檢查座標是否在有效範圍內
            if 0 <= x_img < self.orig_w and 0 <= y_img < self.orig_h:
                # 創建框
                box = [
                    min(self.roi_start[0], x_img),
                    min(self.roi_start[1], y_img),
                    max(self.roi_start[0], x_img),
                    max(self.roi_start[1], y_img)
                ]

                # 获取当前选择的类别
                current_class = self.class_var.get() if self.class_var.get() else "Unknown"

                # 添加到選擇列表，包含类别信息
                self.prompts.append({
                    'bbox': box,
                    'class': current_class
                })

                # 轉換座標到canvas空間進行顯示
                x1_canvas = box[0] / self.scale_x + self.offset_x
                y1_canvas = box[1] / self.scale_y + self.offset_y
                x2_canvas = box[2] / self.scale_x + self.offset_x
                y2_canvas = box[3] / self.scale_y + self.offset_y

                # 繪製永久矩形
                self.canvas.create_rectangle(x1_canvas, y1_canvas, x2_canvas, y2_canvas,
                                            outline='red', width=2)

            self.drawing = False
            self.roi_start = None

            # 清除臨時矩形
            self.canvas.delete("temp_rect")

            # 重新繪製所有邊界框以更新顏色和標籤
            self.redraw_existing_boxes()

    def reset_selections(self):
        self.prompts = []
        self.display_image(self.frame_orig)

    def reselect_video(self):
        """重新選擇視頻檔案"""
        # 選擇新的影片檔案
        new_video_path = filedialog.askopenfilename(
            title="選擇影片檔案",
            initialdir="./test_data/",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v"),
                ("All files", "*.*")
            ]
        )

        if not new_video_path:
            print("未選擇新的視頻檔案")
            return

        # 更新視頻路徑
        self.video_path = new_video_path

        # 重新加載視頻和第一幀
        self.cap.release()  # 釋放當前視頻捕獲對象
        self.cap = cv2.VideoCapture(self.video_path)
        success, self.frame_orig = self.cap.read()

        if not success:
            print("無法讀取新的視頻檔案")
            return

        self.orig_h, self.orig_w = self.frame_orig.shape[:2]

        # 重置所有選擇和狀態
        self.prompts = []
        self.roi_start = None
        self.drawing = False
        self.current_rect = None

        # 重新顯示初始圖像
        self.display_image(self.frame_orig)
        print(f"已更換視頻檔案: {self.video_path}")

    def toggle_save_video(self):
        """切換是否儲存影片的狀態"""
        self.save_video = not self.save_video
        status_text = "是" if self.save_video else "否"
        self.save_video_btn.config(text=f"儲存影片: {status_text}")

    def toggle_save_masks_only(self):
        """切換是否儲存僅有Mask影片的狀態"""
        self.save_masks_only = not self.save_masks_only
        status_text = "是" if self.save_masks_only else "否"
        self.save_masks_only_btn.config(text=f"儲存Mask影片: {status_text}")

    def generate_color_map(self):
        """生成類別顏色映射"""
        # 定義一組預設的RGB顏色
        default_colors = [
            (255, 0, 0),     # red
            (0, 255, 0),     # green
            (0, 0, 255),     # blue
            (255, 255, 0),   # yellow
            (255, 0, 255),   # magenta
            (0, 255, 255),   # cyan
            (255, 165, 0),   # orange
            (128, 0, 128),   # purple
            (42, 42, 165),   # brown
            (203, 192, 255), # pink
            (128, 128, 128), # gray
            (0, 128, 128),   # olive
            (0, 0, 128),     # maroon
            (128, 128, 0)    # teal
        ]

        # 為每個類別分配顏色和透明度，如果沒有定義則使用預設顏色和預設透明度
        for i, class_name in enumerate(self.classes):
            if class_name not in self.color_map:
                color_idx = i % len(default_colors)
                self.color_map[class_name] = default_colors[color_idx]

            # 為每個類別設置預設透明度
            if class_name not in self.alpha_map:
                self.alpha_map[class_name] = 0.5  # 預設透明度為0.5

    def on_class_selected(self, event=None):
        """當下拉選單選擇類別時觸發"""
        # 更新當前類別索引
        selected_class = self.class_var.get()
        if selected_class in self.classes:
            self.current_class_index = self.classes.index(selected_class)

        # 更新RGB輸入框以顯示當前類別的顏色
        if selected_class in self.color_map:
            color = self.color_map[selected_class]
            if isinstance(color, tuple) and len(color) == 3:
                r, g, b = color
                self.r_var.set(str(r))
                self.g_var.set(str(g))
                self.b_var.set(str(b))

        # 更新Alpha輸入框以顯示當前類別的透明度
        if selected_class in self.alpha_map:
            alpha = self.alpha_map[selected_class]
            self.alpha_var.set(str(alpha))

    def update_class_dropdown(self):
        """更新类别下拉菜单"""
        self.class_dropdown['values'] = self.classes
        if self.classes:
            # 如果当前选择的类别仍然存在，则保持不变，否则选择第一个
            if self.current_class_index < len(self.classes):
                self.class_var.set(self.classes[self.current_class_index])
            else:
                self.current_class_index = 0
                self.class_var.set(self.classes[0])
        else:
            self.class_var.set("")

    def add_class(self):
        """添加類別"""
        new_class_name = f"Class_{len(self.classes)+1}"
        self.classes.append(new_class_name)
        # 為新類別生成顏色和透明度
        self.generate_color_map()
        self.class_count_var.set(str(len(self.classes)))
        self.update_class_dropdown()

        # 自動選擇新添加的類別並更新RGB和透明度輸入框
        self.class_var.set(new_class_name)
        self.on_class_selected()

    def remove_class(self):
        """刪除當前選擇的類別"""
        if len(self.classes) > 1:  # 至少保留一個類別
            selected_class = self.class_var.get()
            if not selected_class:
                import tkinter.messagebox
                tkinter.messagebox.showwarning("警告", "請選擇一個類別進行刪除")
                return

            # 從類別列表中移除
            if selected_class in self.classes:
                self.classes.remove(selected_class)

            # 同時移除該類別的所有邊界框
            self.prompts = [prompt for prompt in self.prompts if prompt['class'] != selected_class]
            # 移除該類別的顏色和透明度映射
            if selected_class in self.color_map:
                del self.color_map[selected_class]
            if selected_class in self.alpha_map:
                del self.alpha_map[selected_class]
            # 重新生成顏色映射
            self.generate_color_map()
            self.class_count_var.set(str(len(self.classes)))
            self.update_class_dropdown()
            # 重新顯示圖像以更新顯示
            self.display_image(self.frame_orig)
        else:
            # 如果只有一個類別，可以提示用戶不能刪除
            import tkinter.messagebox
            tkinter.messagebox.showwarning("警告", "至少需要保留一個類別")

    def set_class_color(self):
        """設置當前選擇類別的顏色"""
        selected_class = self.class_var.get()
        if not selected_class:
            import tkinter.messagebox
            tkinter.messagebox.showwarning("警告", "請先選擇一個類別")
            return

        try:
            # 獲取RGB值
            r = int(self.r_var.get())
            g = int(self.g_var.get())
            b = int(self.b_var.get())

            # 驗證RGB值範圍
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                import tkinter.messagebox
                tkinter.messagebox.showerror("錯誤", "RGB值必須在0-255之間")
                return

            # 獲取透明度值
            alpha = float(self.alpha_var.get())

            # 驗證透明度範圍
            if not (0.0 <= alpha <= 1.0):
                import tkinter.messagebox
                tkinter.messagebox.showerror("錯誤", "透明度值必須在0.0-1.0之間")
                return

            # 更新顏色映射
            self.color_map[selected_class] = (r, g, b)
            # 更新透明度映射
            self.alpha_map[selected_class] = alpha

            print(f"類別 '{selected_class}' 的顏色已設置為 RGB({r}, {g}, {b}), 透明度: {alpha}")

            # 立即刷新顯示，更新所有相同類別的框的顏色
            self.display_image(self.frame_orig)

            # 立即儲存配置
            self.save_config()

        except ValueError:
            import tkinter.messagebox
            tkinter.messagebox.showerror("錯誤", "請輸入有效的數字")

    def save_config(self):
        """儲存配置檔案"""
        try:
            config = {
                'classes': self.classes,
                'color_map': {class_name: list(color) for class_name, color in self.color_map.items()},
                'alpha_map': self.alpha_map
            }

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            print(f"配置已儲存: {self.config_path}")

        except Exception as e:
            print(f"儲存配置檔案時出錯: {e}")

    def rename_class(self):
        """重命名當前選擇的類別"""
        import tkinter.simpledialog

        # 獲取當前選擇的類別
        selected_class = self.class_var.get()
        if not selected_class:
            import tkinter.messagebox
            tkinter.messagebox.showwarning("警告", "請先選擇一個類別")
            return

        # 獲取類別在列表中的索引
        try:
            idx = self.classes.index(selected_class)
        except ValueError:
            import tkinter.messagebox
            tkinter.messagebox.showerror("錯誤", "找不到所選類別")
            return

        # 彈出對話框讓用戶輸入新名稱
        new_name = tkinter.simpledialog.askstring(
            "重命名類別",
            f"請輸入新的類別名稱:",
            initialvalue=selected_class
        )

        if new_name is not None and new_name.strip():  # 確保輸入不為空
            # 更新類別列表
            old_name = self.classes[idx]
            self.classes[idx] = new_name.strip()

            # 更新顏色映射
            color = self.color_map.get(old_name, (255, 0, 0))  # 保留原來的顏色
            alpha = self.alpha_map.get(old_name, 0.5)  # 保留原來的透明度
            del self.color_map[old_name]  # 刪除舊的映射
            del self.alpha_map[old_name]  # 刪除舊的透明度映射
            self.color_map[new_name.strip()] = color  # 添加新的顏色映射
            self.alpha_map[new_name.strip()] = alpha  # 添加新的透明度映射

            # 更新所有屬於該類別的prompt
            for prompt in self.prompts:
                if prompt['class'] == old_name:
                    prompt['class'] = new_name.strip()

            self.update_class_dropdown()
            # 重新顯示圖像以更新顯示
            self.display_image(self.frame_orig)

    def start_tracking(self):
        if not self.prompts:
            print("請先選擇至少一個區域")
            return

        print(f"已選擇 {len(self.prompts)} 個區域: {self.prompts}")

        # 釋放視頻捕獲對象
        self.cap.release()

        # 隱藏主窗口
        self.root.withdraw()

        # 開始視頻追蹤
        bboxes_for_tracking = []
        for prompt in self.prompts:
            bboxes_for_tracking.append(prompt['bbox'])  # 只提取bbox部分

        # 創建新的窗口顯示追蹤結果
        tracking_window = tk.Toplevel(self.root)
        tracking_window.title("SAM2 Video Tracking Results")
        tracking_window.geometry("800x600")

        # 創建按鈕框架
        button_frame = ttk.Frame(tracking_window)
        button_frame.pack(fill=tk.X, pady=(5, 0))

        # 停止追蹤按鈕
        stop_btn = ttk.Button(button_frame, text="停止追蹤", command=lambda: stop_tracking())
        stop_btn.pack(side=tk.LEFT, padx=5, pady=5)

        tracking_canvas = tk.Canvas(tracking_window, bg='black')
        tracking_canvas.pack(fill=tk.BOTH, expand=True)

        # 保存對象引用以避免被垃圾回收
        self.tracking_canvas = tracking_canvas

        # 追蹤是否停止的標誌
        self.tracking_stopped = False

        # 視頻寫入器初始化
        video_writer = None
        mask_video_writer = None
        self.output_path = None  # 存儲輸出路徑以便在其他函數中訪問
        self.mask_output_path = None  # 存儲Mask視頻輸出路徑

        if self.save_video:
            import os
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)

            # 獲取視頻的基本信息
            cap_info = cv2.VideoCapture(self.video_path)
            fps = int(cap_info.get(cv2.CAP_PROP_FPS))
            width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_info.release()

            # 生成輸出視頻路徑
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_path = os.path.join(output_dir, f"tracking_result_{timestamp}.mp4")

            # 初始化視頻寫入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

            print(f"開始儲存視頻到: {self.output_path}")

        if self.save_masks_only:
            import os
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)

            # 獲取視頻的基本信息
            cap_info = cv2.VideoCapture(self.video_path)
            fps = int(cap_info.get(cv2.CAP_PROP_FPS))
            width = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_info.release()

            # 生成輸出Mask視頻路徑
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.mask_output_path = os.path.join(output_dir, f"mask_result_{timestamp}.mp4")

            # 初始化Mask視頻寫入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            mask_video_writer = cv2.VideoWriter(self.mask_output_path, fourcc, fps, (width, height))

            print(f"開始儲存Mask視頻到: {self.mask_output_path}")

        # 創建新的predictor實例以確保每次都能正確開始
        overrides = self.base_overrides.copy()
        # 移除原本的儲存設置，因為我們將手動處理
        if 'save' in overrides:
            del overrides['save']
        if 'project' in overrides:
            del overrides['project']

        fresh_predictor = SAM2VideoPredictor(overrides=overrides)

        try:
            results = fresh_predictor(
                source=self.video_path,
                bboxes=bboxes_for_tracking,
                stream=True
            )
        except Exception as e:
            print(f"初始化追蹤時出錯: {e}")
            tracking_window.destroy()
            self.root.deiconify()  # 重新顯示主視窗
            return

        def stop_tracking():
            """停止追蹤並關閉視窗"""
            self.tracking_stopped = True
            # 釋放視頻寫入器
            if video_writer is not None:
                video_writer.release()
                if self.output_path:
                    print(f"視頻已儲存完成: {self.output_path}")
                else:
                    print("視頻已儲存完成")
            # 釋放Mask視頻寫入器
            if mask_video_writer is not None:
                mask_video_writer.release()
                if self.mask_output_path:
                    print(f"Mask視頻已儲存完成: {self.mask_output_path}")
                else:
                    print("Mask視頻已儲存完成")
            tracking_window.destroy()
            self.root.deiconify()  # 重新顯示主視窗

        def update_frame():
            if self.tracking_stopped:
                return

            try:
                result = next(results)

                # 使用原始圖像作為基礎
                annotated_frame = result.orig_img.copy()

                # 獲取分割結果
                masks = result.masks
                boxes = result.boxes

                # 如果有分割結果，則繪製
                if masks is not None:
                    # 繪製分割掩碼
                    for i, mask in enumerate(masks):
                        # 獲取對應的類別索引
                        if boxes is not None and i < len(boxes):
                            cls_id = int(boxes[i].cls[0]) if len(boxes[i].cls) > 0 else 0

                            # 使用我們自定義的類別名稱
                            if i < len(self.prompts):
                                class_name = self.prompts[i]['class']

                                # 獲取該類別的顏色
                                color = self.color_map.get(class_name, (255, 0, 0))  # 默認為紅色 (BGR)
                                # 直接使用RGB元組，轉換為BGR格式 (OpenCV使用BGR順序)
                                if isinstance(color, tuple) and len(color) == 3:
                                    # 將RGB轉換為BGR (r,g,b -> b,g,r)
                                    color_bgr = (color[2], color[1], color[0])
                                else:
                                    color_bgr = (0, 0, 255)  # 默認為紅色 (BGR)

                                # 獲取該類別的透明度
                                alpha = self.alpha_map.get(class_name, 0.5)  # 默認透明度為0.5

                                # 繪製分割掩碼
                                mask_np = mask.data.cpu().numpy()[0]  # 轉換為numpy數組
                                mask_uint8 = (mask_np * 255).astype(np.uint8)  # 轉換為uint8格式

                                # 將掩碼應用到圖像上，使用指定的透明度
                                colored_mask = np.zeros_like(annotated_frame)
                                colored_mask[:] = color_bgr

                                # 應用透明度混合
                                mask_binary = mask_np > 0.5
                                annotated_frame[mask_binary] = (
                                    annotated_frame[mask_binary] * (1 - alpha) +
                                    colored_mask[mask_binary] * alpha
                                ).astype(np.uint8)

                                # 可選：繪製邊界框（根據需求決定是否顯示）
                                # 如果只需要顯示mask而不顯示邊界框，可以註釋掉下面的代碼
                                # if boxes is not None and i < len(boxes):
                                #     box = boxes[i].xyxy[0].cpu().numpy()
                                #     x1, y1, x2, y2 = map(int, box)

                                #     # 繪製邊界框
                                #     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)

                                #     # 在邊界框上方繪製類別名稱
                                #     cv2.putText(annotated_frame, class_name, (x1, y1 - 10),
                                #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

                # 如果需要保存視頻，將當前幀寫入視頻文件
                if video_writer is not None:
                    video_writer.write(annotated_frame)

                # 如果需要保存Mask視頻，生成純Mask幀並寫入視頻文件
                if mask_video_writer is not None:
                    # 創建黑色背景的Mask幀
                    mask_frame = np.zeros_like(annotated_frame)

                    # 繪製分割掩碼到純黑背景上
                    if masks is not None:
                        for i, mask in enumerate(masks):
                            if i < len(self.prompts):
                                class_name = self.prompts[i]['class']

                                # 獲取該類別的顏色
                                color = self.color_map.get(class_name, (255, 0, 0))  # 默認為紅色 (BGR)
                                # 直接使用RGB元組，轉換為BGR格式 (OpenCV使用BGR順序)
                                if isinstance(color, tuple) and len(color) == 3:
                                    # 將RGB轉換為BGR (r,g,b -> b,g,r)
                                    color_bgr = (color[2], color[1], color[0])
                                else:
                                    color_bgr = (0, 0, 255)  # 默認為紅色 (BGR)

                                # 獲取該類別的透明度
                                alpha = self.alpha_map.get(class_name, 0.5)  # 默認透明度為0.5

                                # 繪製分割掩碼
                                mask_np = mask.data.cpu().numpy()[0]  # 轉換為numpy數組
                                mask_uint8 = (mask_np * 255).astype(np.uint8)  # 轉換為uint8格式

                                # 應用透明度混合到黑色背景
                                mask_binary = mask_np > 0.5
                                mask_frame[mask_binary] = (
                                    mask_frame[mask_binary] * (1 - alpha) +
                                    np.full_like(mask_frame[mask_binary], color_bgr) * alpha
                                ).astype(np.uint8)

                    mask_video_writer.write(mask_frame)

                # 轉換BGR到RGB
                image_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # 獲取當前canvas大小
                canvas_width = tracking_canvas.winfo_width()
                canvas_height = tracking_canvas.winfo_height()

                # 如果canvas尺寸未初始化，使用默認值
                if canvas_width <= 1 or canvas_height <= 1:
                    canvas_width = 800
                    canvas_height = 600

                # 計算縮放比例以保持縱橫比
                h, w = image_rgb.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)

                # 調整圖像大小
                resized_image = cv2.resize(image_rgb, (new_w, new_h))

                # 轉換為PIL Image
                pil_image = Image.fromarray(resized_image)
                photo = ImageTk.PhotoImage(pil_image)

                # 在canvas上顯示圖像
                tracking_canvas.delete("all")
                tracking_canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)

                # 保持對photo的引用以避免被垃圾回收
                tracking_canvas.image = photo

                # 繼續下一幀
                if not self.tracking_stopped:
                    tracking_window.after(30, update_frame)  # 大約33fps

            except StopIteration:
                print("視頻播放完畢")
                # 釋放視頻寫入器
                if video_writer is not None:
                    video_writer.release()
                    if self.output_path:
                        print(f"視頻已儲存完成: {self.output_path}")
                    else:
                        print("視頻已儲存完成")
                # 釋放Mask視頻寫入器
                if mask_video_writer is not None:
                    mask_video_writer.release()
                    if self.mask_output_path:
                        print(f"Mask視頻已儲存完成: {self.mask_output_path}")
                    else:
                        print("Mask視頻已儲存完成")
                if not self.tracking_stopped:
                    tracking_window.destroy()
                    self.root.deiconify()  # 重新顯示主視窗
            except Exception as e:
                # 釋放視頻寫入器
                if video_writer is not None:
                    video_writer.release()
                    if self.output_path:
                        print(f"視頻已儲存完成: {self.output_path}")
                    else:
                        print("視頻已儲存完成")
                # 釋放Mask視頻寫入器
                if mask_video_writer is not None:
                    mask_video_writer.release()
                    if self.mask_output_path:
                        print(f"Mask視頻已儲存完成: {self.mask_output_path}")
                    else:
                        print("Mask視頻已儲存完成")
                if not self.tracking_stopped:
                    print(f"追蹤過程中出錯: {e}")
                    tracking_window.destroy()
                    self.root.deiconify()  # 重新顯示主視窗

        # 在開始追蹤前儲存配置
        self.save_config()

        # 開始顯示第一幀
        update_frame()

        # 綁定關閉事件
        def on_closing():
            self.tracking_stopped = True
            # 釋放視頻寫入器
            if video_writer is not None:
                video_writer.release()
                if self.output_path:
                    print(f"視頻已儲存完成: {self.output_path}")
                else:
                    print("視頻已儲存完成")
            # 釋放Mask視頻寫入器
            if mask_video_writer is not None:
                mask_video_writer.release()
                if self.mask_output_path:
                    print(f"Mask視頻已儲存完成: {self.mask_output_path}")
                else:
                    print("Mask視頻已儲存完成")
            tracking_window.destroy()
            self.root.deiconify()  # 重新顯示主視窗

        tracking_window.protocol("WM_DELETE_WINDOW", on_closing)


def main():
    root = tk.Tk()
    app = SAM2TrackerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()