import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from ultralytics.models.sam import SAM2VideoPredictor

class SAM2TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM2 Video Tracker")

        # 初始化變量
        self.prompts = []
        self.roi_start = None
        self.drawing = False
        self.current_rect = None
        self.save_video = False  # 是否保存视频的标志
        self.classes = ["Object"]  # 类别列表，默认为"Object"
        self.current_class_index = 0  # 当前选择的类别索引

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
            print("無法讀取視頻文件")
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

    def setup_gui(self):
        # 主容器
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 圖像顯示區域
        self.canvas = tk.Canvas(main_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 綁定鼠標事件
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # 綁定窗口大小改變事件
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # 按鈕框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        # 完成選擇按鈕
        self.finish_btn = ttk.Button(button_frame, text="完成選擇並開始追蹤", command=self.start_tracking)
        self.finish_btn.pack(side=tk.LEFT, padx=(0, 10))

        # 重置按鈕
        self.reset_btn = ttk.Button(button_frame, text="重置選擇", command=self.reset_selections)
        self.reset_btn.pack(side=tk.LEFT)

        # 保存影片開關按鈕
        self.save_video_btn = ttk.Button(button_frame, text="保存影片: 否", command=self.toggle_save_video)
        self.save_video_btn.pack(side=tk.RIGHT)

        # 类别控制框架
        class_control_frame = ttk.Frame(main_frame)
        class_control_frame.pack(fill=tk.X, pady=(5, 0))

        # 类别数量控制
        ttk.Label(class_control_frame, text="类别数量:").pack(side=tk.LEFT, padx=(0, 5))
        self.class_count_var = tk.StringVar(value=str(len(self.classes)))
        self.class_count_label = ttk.Label(class_control_frame, textvariable=self.class_count_var)
        self.class_count_label.pack(side=tk.LEFT, padx=(0, 10))

        self.add_class_btn = ttk.Button(class_control_frame, text="+", width=3, command=self.add_class)
        self.add_class_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.remove_class_btn = ttk.Button(class_control_frame, text="-", width=3, command=self.remove_class)
        self.remove_class_btn.pack(side=tk.LEFT, padx=(0, 10))

        # 类别选择下拉菜单
        ttk.Label(class_control_frame, text="当前类别:").pack(side=tk.LEFT, padx=(0, 5))
        self.class_var = tk.StringVar()
        self.class_dropdown = ttk.Combobox(class_control_frame, textvariable=self.class_var, state="readonly")
        self.class_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        self.update_class_dropdown()

        # 类别命名按钮
        self.rename_class_btn = ttk.Button(class_control_frame, text="重命名类别", command=self.rename_class)
        self.rename_class_btn.pack(side=tk.LEFT)

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

            self.canvas.create_rectangle(x1_canvas, y1_canvas, x2_canvas, y2_canvas,
                                        outline='red', width=2)
            # 显示类别标签
            self.canvas.create_text(x1_canvas, y1_canvas - 10, text=class_name,
                                   fill='red', anchor='sw', font=('Arial', 10, 'bold'))

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

    def reset_selections(self):
        self.prompts = []
        self.display_image(self.frame_orig)

    def toggle_save_video(self):
        """切換是否保存影片的狀態"""
        self.save_video = not self.save_video
        status_text = "是" if self.save_video else "否"
        self.save_video_btn.config(text=f"保存影片: {status_text}")

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
        """添加类别"""
        new_class_name = f"Class_{len(self.classes)+1}"
        self.classes.append(new_class_name)
        self.class_count_var.set(str(len(self.classes)))
        self.update_class_dropdown()

    def remove_class(self):
        """删除最后一个类别"""
        if len(self.classes) > 1:  # 至少保留一个类别
            self.classes.pop()
            self.class_count_var.set(str(len(self.classes)))
            self.update_class_dropdown()
        else:
            # 如果只有一个类别，可以提示用户不能删除
            import tkinter.messagebox
            tkinter.messagebox.showwarning("警告", "至少需要保留一个类别")

    def rename_class(self):
        """重命名当前选择的类别"""
        import tkinter.simpledialog

        # 获取当前选择的类别
        selected_class = self.class_var.get()
        if not selected_class:
            import tkinter.messagebox
            tkinter.messagebox.showwarning("警告", "请先选择一个类别")
            return

        # 获取类别在列表中的索引
        try:
            idx = self.classes.index(selected_class)
        except ValueError:
            import tkinter.messagebox
            tkinter.messagebox.showerror("错误", "找不到所选类别")
            return

        # 弹出对话框让用户输入新名称
        new_name = tkinter.simpledialog.askstring(
            "重命名类别",
            f"请输入新的类别名称:",
            initialvalue=selected_class
        )

        if new_name is not None and new_name.strip():  # 确保输入不为空
            self.classes[idx] = new_name.strip()
            self.update_class_dropdown()

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

        # 創建新的predictor實例以確保每次都能正確開始
        overrides = self.base_overrides.copy()
        if self.save_video:
            import os
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            overrides['save'] = True
            overrides['project'] = output_dir

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
            self.root.deiconify()  # 重新顯示主窗口
            return

        def stop_tracking():
            """停止追蹤並關閉視窗"""
            self.tracking_stopped = True
            tracking_window.destroy()
            self.root.deiconify()  # 重新顯示主窗口

        def update_frame():
            if self.tracking_stopped:
                return

            try:
                result = next(results)
                annotated_frame = result.plot()  # 使用Ultralytics的plot方法直接獲取標註後的幀

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
                if not self.tracking_stopped:
                    tracking_window.destroy()
                    self.root.deiconify()  # 重新顯示主窗口
            except Exception as e:
                if not self.tracking_stopped:
                    print(f"追蹤過程中出錯: {e}")
                    tracking_window.destroy()
                    self.root.deiconify()  # 重新顯示主窗口

        # 開始顯示第一幀
        update_frame()

        # 綁定關閉事件
        def on_closing():
            self.tracking_stopped = True
            tracking_window.destroy()
            self.root.deiconify()  # 重新顯示主窗口

        tracking_window.protocol("WM_DELETE_WINDOW", on_closing)


def main():
    root = tk.Tk()
    app = SAM2TrackerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
