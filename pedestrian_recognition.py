import cv2
import numpy as np
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QCheckBox, 
                            QFileDialog, QMessageBox, QListWidget, QSplitter)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import imutils
from PIL import Image, ImageDraw, ImageFont

class PedestrianRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化变量
        self.camera = None
        self.video_path = 'video'
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        self.is_paused = False
        self.traffic_light_detection_enabled = False
        self.traffic_light_state = None  # 'red', 'green', or None
        self.traffic_light_class_id = 9  # 红绿灯在COCO数据集中的类别ID
        
        # 处理PyInstaller打包后的路径问题
        if hasattr(sys, '_MEIPASS'):
            # 在打包后的环境中
            self.base_dir = sys._MEIPASS
            
            # 优先使用exe文件所在目录下的video文件夹，方便用户添加自己的视频
            exe_dir = os.path.dirname(os.path.abspath(sys.executable))
            self.default_video_folder = os.path.join(exe_dir, "video")
        else:
            # 在开发环境中
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
            self.default_video_folder = os.path.join(self.base_dir, "video")
        
        # 确保video文件夹存在
        os.makedirs(self.default_video_folder, exist_ok=True)
        # 选中的视频文件路径
        self.selected_video_path = None
        
        # 加载YOLOv3模型 (使用OpenCV dnn模块，避免PyTorch依赖)
        self.net = None
        self.output_layers = []
        self.classes = []
        self.person_class_id = 0
        self.load_yolo_model()
        
        # 初始化界面
        self.init_ui()
        
        # 尝试加载中文字体，使用默认字体作为备选
        try:
            # 尝试使用系统中的SimHei字体
            self.font = ImageFont.truetype("simhei.ttf", 20)
        except:
            # 如果找不到指定字体，使用PIL默认字体
            self.font = ImageFont.load_default()
        

    
    def load_default_video(self):
        """从默认视频文件夹加载第一个视频文件"""
        try:
            # 确保默认视频文件夹存在
            os.makedirs(self.default_video_folder, exist_ok=True)
            
            # 获取默认文件夹中的所有视频文件
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = []
            
            for file in os.listdir(self.default_video_folder):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(self.default_video_folder, file))
            
            # 如果找到视频文件，自动加载第一个
            if video_files:
                # 按文件名排序，确保加载的是确定性的第一个文件
                video_files.sort()
                self.video_path = video_files[0]
                self.status_label.setText(f'已从默认文件夹加载视频: {os.path.basename(self.video_path)}')
        except Exception as e:
            # 如果自动加载失败，不影响程序运行
            print(f"自动加载默认视频失败: {e}")
    

    
    def load_yolo_model(self):
        """加载YOLOv3模型配置和权重"""
        # 模型配置文件和权重文件的URL
        cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
        weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
        names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        
        # 确定模型保存路径
        if hasattr(sys, '_MEIPASS'):
            # 在打包后的环境中，优先使用exe文件所在目录
            exe_dir = os.path.dirname(os.path.abspath(sys.executable))
            model_dir = os.path.join(exe_dir, "yolo_models")
        else:
            # 在开发环境中
            model_dir = os.path.join(self.base_dir, "yolo_models")
        
        # 确保文件夹存在
        os.makedirs(model_dir, exist_ok=True)
        
        cfg_path = os.path.join(model_dir, "yolov3-tiny.cfg")
        weights_path = os.path.join(model_dir, "yolov3-tiny.weights")
        names_path = os.path.join(model_dir, "coco.names")
        
        # 如果文件不存在，则下载
        if not os.path.exists(cfg_path):
            print(f"正在下载YOLO配置文件到 {cfg_path}")
            self.download_file(cfg_url, cfg_path)
        
        if not os.path.exists(weights_path):
            print(f"正在下载YOLO权重文件到 {weights_path}")
            print("注意：这可能需要一些时间，请耐心等待...")
            self.download_file(weights_url, weights_path)
        
        if not os.path.exists(names_path):
            print(f"正在下载类别名称文件到 {names_path}")
            self.download_file(names_url, names_path)
        
        # 加载类别名称
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # 加载模型
        try:
            self.net = cv2.dnn.readNet(weights_path, cfg_path)
            # 设置计算后端
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # 获取输出层
            layer_names = self.net.getLayerNames()
            # OpenCV 4.x和3.x获取输出层的方式不同
            try:
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            except:
                self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            
            print("YOLO模型加载成功")
        except Exception as e:
            print(f"加载YOLO模型时出错: {e}")
            print("将使用默认的行人检测方法")
    
    def download_file(self, url, save_path):
        """下载文件的辅助方法，带进度条、重试机制和多线程下载支持"""
        import urllib.request
        import time
        import sys
        import threading
        from concurrent.futures import ThreadPoolExecutor
        import shutil
        import os
        
        # 替代下载源列表
        alternative_urls = {
            "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg?raw=true": [
                "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
                "https://pjreddie.com/media/files/yolov3-tiny.cfg"
            ],
            "https://pjreddie.com/media/files/yolov3-tiny.weights": [
                "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3-tiny.weights"
            ],
            "https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true": [
                "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
            ]
        }
        
        # 获取所有可能的下载源
        all_urls = [url] + alternative_urls.get(url, [])
        max_retries = 3
        retry_count = 0
        current_url_index = 0
        
        # 尝试所有下载源
        while current_url_index < len(all_urls) and retry_count < max_retries:
            current_url = all_urls[current_url_index]
            try:
                print(f"正在下载 {os.path.basename(save_path)} (尝试 {retry_count + 1}/{max_retries}, 源 {current_url_index + 1}/{len(all_urls)})...")
                print(f"下载地址: {current_url}")
                
                # 检查是否支持范围请求
                req = urllib.request.Request(current_url, method='HEAD')
                with urllib.request.urlopen(req) as response:
                    content_length = response.getheader('Content-Length')
                    accept_ranges = response.getheader('Accept-Ranges')
                
                total_size = int(content_length) if content_length else 0
                
                # 回调函数用于显示下载进度
                class ProgressTracker:
                    def __init__(self):
                        self.total_downloaded = 0
                        self.lock = threading.Lock()
                        self.start_time = time.time()
                    
                    def update(self, downloaded):
                        with self.lock:
                            self.total_downloaded += downloaded
                            if total_size > 0:
                                percent = min(100, int(self.total_downloaded * 100 / total_size))
                                elapsed = time.time() - self.start_time
                                if elapsed > 0:
                                    speed = self.total_downloaded / elapsed / 1024  # KB/s
                                else:
                                    speed = 0
                                sys.stdout.write(f"\r[{percent:3d}%] 已下载: {self.total_downloaded / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB | 速度: {speed:.2f} KB/s")
                                sys.stdout.flush()
                
                tracker = ProgressTracker()
                
                # 如果服务器支持范围请求且文件较大，使用多线程下载
                if accept_ranges == 'bytes' and total_size > 10 * 1024 * 1024:  # 文件大于10MB
                    print(f"文件较大，启用多线程下载模式")
                    self._download_multithreaded(current_url, save_path, total_size, tracker)
                else:
                    # 使用单线程下载
                    def progress_hook(count, block_size, total_size):
                        tracker.update(block_size)
                    
                    urllib.request.urlretrieve(current_url, save_path, reporthook=progress_hook)
                
                print(f"\n下载完成: {save_path}")
                return  # 下载成功，退出函数
                
            except Exception as e:
                retry_count += 1
                print(f"\n下载失败: {e}")
                if retry_count >= max_retries:
                    retry_count = 0
                    current_url_index += 1
                else:
                    wait_time = 2 * retry_count  # 指数退避
                    print(f"{wait_time}秒后重试...")
                    time.sleep(wait_time)
        
        print(f"所有下载源均失败，请手动下载 {url} 并保存为 {save_path}")
    
    def _download_multithreaded(self, url, save_path, total_size, tracker):
        """多线程下载函数"""
        import urllib.request
        import tempfile
        import os
        
        num_threads = 4  # 使用4个线程下载
        chunk_size = total_size // num_threads
        temp_files = []
        
        def download_chunk(start, end, temp_file):
            """下载文件的一个块"""
            headers = {'Range': f'bytes={start}-{end}'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response, open(temp_file, 'wb') as f:
                while True:
                    data = response.read(8192)  # 8KB chunks
                    if not data:
                        break
                    f.write(data)
                    tracker.update(len(data))
        
        try:
            # 创建临时目录
            temp_dir = tempfile.mkdtemp()
            
            # 创建线程池
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = []
                # 提交下载任务
                for i in range(num_threads):
                    start = i * chunk_size
                    # 最后一个块下载到文件末尾
                    end = total_size - 1 if i == num_threads - 1 else (i + 1) * chunk_size - 1
                    temp_file = os.path.join(temp_dir, f'chunk_{i}.part')
                    temp_files.append(temp_file)
                    futures.append(executor.submit(download_chunk, start, end, temp_file))
                
                # 等待所有线程完成
                for future in futures:
                    future.result()
            
            # 合并所有块
            with open(save_path, 'wb') as outfile:
                for temp_file in temp_files:
                    with open(temp_file, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile)
        
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            # 清理临时目录
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
    
    def create_video_list_widget(self):
        # 创建视频文件列表组件
        video_list_container = QWidget()
        video_list_layout = QVBoxLayout(video_list_container)
        
        # 标题
        title_label = QLabel("可用视频文件")
        title_label.setStyleSheet("font-weight: bold; color: #333;")
        video_list_layout.addWidget(title_label)
        
        # 视频文件列表
        self.video_list_widget = QListWidget()
        self.video_list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.video_list_widget.itemClicked.connect(self.on_video_item_clicked)
        video_list_layout.addWidget(self.video_list_widget)
        
        # 刷新按钮
        refresh_button = QPushButton("刷新列表")
        refresh_button.clicked.connect(self.refresh_video_list)
        video_list_layout.addWidget(refresh_button)
        
        return video_list_container
    
    def refresh_video_list(self):
        """加载并显示默认视频文件夹中的视频文件"""
        try:
            # 清空当前列表
            self.video_list_widget.clear()
            
            # 确保默认视频文件夹存在
            os.makedirs(self.default_video_folder, exist_ok=True)
            
            # 获取文件夹中的所有视频文件
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = []
            
            for file in os.listdir(self.default_video_folder):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(file)
            
            # 按名称排序
            video_files.sort()
            
            # 添加到列表中
            for file in video_files:
                self.video_list_widget.addItem(file)
                
            self.status_label.setText(f'发现 {len(video_files)} 个视频文件')
        except Exception as e:
            print(f"刷新视频列表失败: {e}")
            self.status_label.setText('刷新视频列表失败')
    
    def on_video_item_clicked(self, item):
        """处理视频列表项点击事件"""
        # 获取选中的视频文件名
        video_filename = item.text()
        # 构建完整路径
        self.selected_video_path = os.path.join(self.default_video_folder, video_filename)
        # 更新状态显示
        self.status_label.setText(f'已选择: {video_filename}，请点击"播放视频文件"按钮开始识别')
        # 如果视频正在播放，停止它
        if self.is_running:
            self.stop_video()
        
    def init_ui(self):
        # 设置窗口标题和大小
        self.setWindowTitle('行人识别与闯红灯检测系统')
        self.setGeometry(100, 100, 1000, 700)
        
        # 创建主布局
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 创建左侧视频文件列表并获取返回的widget
        video_list_widget = self.create_video_list_widget()
        
        # 创建右侧视频显示区域
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 创建视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        right_layout.addWidget(self.video_label)
        
        # 将左侧列表和右侧视频区域添加到分割器
        splitter.addWidget(video_list_widget)
        splitter.addWidget(right_widget)
        
        # 设置分割器的初始大小比例
        splitter.setSizes([200, 800])
        
        # 将分割器添加到主布局
        main_layout.addWidget(splitter)
        
        # 创建控制按钮布局
        control_layout = QHBoxLayout()
        
        # 摄像头按钮
        self.camera_button = QPushButton('使用摄像头')
        self.camera_button.clicked.connect(self.start_camera)
        control_layout.addWidget(self.camera_button)
        
        # 视频文件按钮
        self.video_button = QPushButton('播放视频文件')
        self.video_button.clicked.connect(self.open_video_file)
        control_layout.addWidget(self.video_button)
        
        # 停止按钮
        self.stop_button = QPushButton('停止')
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        # 暂停/继续按钮
        self.pause_button = QPushButton('暂停')
        self.pause_button.clicked.connect(self.toggle_pause)
        self.pause_button.setEnabled(False)
        control_layout.addWidget(self.pause_button)
        
        # 红绿灯检测复选框
        self.traffic_light_checkbox = QCheckBox('开启闯红灯检测')
        self.traffic_light_checkbox.stateChanged.connect(self.toggle_traffic_light_detection)
        control_layout.addWidget(self.traffic_light_checkbox)
        
        main_layout.addLayout(control_layout)
        
        # 添加状态标签
        self.status_label = QLabel('就绪')
        main_layout.addWidget(self.status_label)
        
        # 在status_label初始化后再调用refresh_video_list
        self.refresh_video_list()
    
    def start_camera(self):
        # 停止之前的视频流
        self.stop_video()
        
        # 尝试打开摄像头
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            QMessageBox.critical(self, '错误', '无法打开摄像头')
            return
        
        self.is_running = True
        self.is_paused = False
        self.update_ui_state()
        self.timer.start(30)  # 约33fps
        self.status_label.setText('摄像头运行中...')
    
    def open_video_file(self):
        # 停止之前的视频流
        self.stop_video()
        
        # 如果已经从列表中选择了视频文件，直接使用它
        if self.selected_video_path and os.path.exists(self.selected_video_path):
            file_path = self.selected_video_path
        else:
            # 确保默认视频文件夹存在
            os.makedirs(self.default_video_folder, exist_ok=True)
            
            # 打开文件选择对话框，默认路径设为default_video_folder
            file_path, _ = QFileDialog.getOpenFileName(self, '选择视频文件', self.default_video_folder, 'Video Files (*.mp4 *.avi *.mov *.mkv)')
            if not file_path:
                return
        
        self.video_path = file_path
        self.camera = cv2.VideoCapture(file_path)
        if not self.camera.isOpened():
            QMessageBox.critical(self, '错误', '无法打开视频文件')
            return
        
        self.is_running = True
        self.is_paused = False
        self.update_ui_state()
        self.timer.start(30)
        # 显示更简洁的状态信息
        self.status_label.setText(f'视频播放中: {os.path.basename(file_path)} - 正在进行行人识别...')
    
    def stop_video(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.is_running = False
        self.is_paused = False
        self.update_ui_state()
        self.status_label.setText('已停止')
    
    def toggle_traffic_light_detection(self, state):
        self.traffic_light_detection_enabled = state == Qt.Checked
        if self.traffic_light_detection_enabled:
            self.status_label.setText('已开启闯红灯检测')
        else:
            self.status_label.setText('已关闭闯红灯检测')
    
    def update_ui_state(self):
        self.camera_button.setEnabled(not self.is_running)
        self.video_button.setEnabled(not self.is_running)
        self.stop_button.setEnabled(self.is_running)
        self.pause_button.setEnabled(self.is_running)
    
    def detect_pedestrians(self, frame):
        """使用YOLOv3模型检测行人"""
        # 调整帧大小以平衡性能和精度
        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        
        # 如果YOLO模型没有加载成功，返回空列表
        if self.net is None:
            # 这里可以添加回退到HOG+SVM的逻辑，但暂时返回空
            return frame, []
        
        # 获取图像尺寸
        height, width, channels = frame.shape
        
        # 创建blob对象 (缩放为416x416，归一化，交换通道)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # 将blob输入到网络
        self.net.setInput(blob)
        
        # 获取输出层的结果
        outs = self.net.forward(self.output_layers)
        
        # 存储检测到的行人框
        pedestrian_boxes = []
        
        # 解析输出结果
        for out in outs:
            for detection in out:
                # 获取所有类别的分数
                scores = detection[5:]
                # 获取最高分数的类别ID
                class_id = np.argmax(scores)
                # 获取置信度
                confidence = scores[class_id]
                
                # 如果是人类且置信度大于阈值
                if class_id == self.person_class_id and confidence > 0.6:
                    # 计算边界框坐标 (YOLO输出的是中心点和宽高)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # 计算左上角坐标
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # 确保坐标在图像范围内
                    x = max(0, x)
                    y = max(0, y)
                    w = min(width - x, w)
                    h = min(height - y, h)
                    
                    # 过滤掉过小的检测框，避免误检
                    if w > 30 and h > 60:
                        pedestrian_boxes.append((x, y, w, h))
                        # 绘制绿色矩形框
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 对重叠的检测框应用非最大值抑制，提高检测精度
        if len(pedestrian_boxes) > 0:
            # 转换为OpenCV需要的格式
            boxes = np.array(pedestrian_boxes)
            # 创建置信度数组 (这里简化处理，全部设为1)
            confidences = np.ones(len(boxes))
            # 应用非最大值抑制
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), 0.4, 0.4)
            # 只保留非最大值抑制后的框
            if len(indices) > 0:
                # 根据OpenCV版本处理不同格式的indices
                filtered_boxes = []
                if isinstance(indices, np.ndarray):
                    # 处理不同版本的OpenCV返回格式
                    if indices.ndim == 2:
                        # 格式为 [[0], [1], [2]]
                        for i in indices:
                            idx = i[0]
                            filtered_boxes.append(pedestrian_boxes[idx])
                            # 重新绘制保留的矩形框
                            x, y, w, h = pedestrian_boxes[idx]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        # 格式为 [0, 1, 2]
                        for i in indices:
                            idx = int(i)
                            filtered_boxes.append(pedestrian_boxes[idx])
                            # 重新绘制保留的矩形框
                            x, y, w, h = pedestrian_boxes[idx]
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    # 兼容列表格式
                    for i in indices:
                        if isinstance(i, list) or isinstance(i, tuple):
                            idx = i[0]
                        else:
                            idx = i
                        filtered_boxes.append(pedestrian_boxes[idx])
                        # 重新绘制保留的矩形框
                        x, y, w, h = pedestrian_boxes[idx]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                pedestrian_boxes = filtered_boxes
        
        return frame, pedestrian_boxes
    
    def detect_traffic_light(self, frame):
        """使用YOLOv3检测红绿灯位置，然后通过颜色分析判断状态"""
        # 首先使用YOLO检测红绿灯位置
        traffic_light_boxes = []
        traffic_light_state = "未知"
        
        # 如果YOLO模型没有加载成功，直接返回原帧和未知状态
        if self.net is None:
            frame = self.draw_chinese_text(frame, "红绿灯: 检测模型未加载", (10, 30), font_size=20, color=(255, 255, 255))
            return frame, traffic_light_state
        
        # 获取图像尺寸
        height, width, channels = frame.shape
        
        # 创建blob对象
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # 将blob输入到网络
        self.net.setInput(blob)
        
        # 获取输出层的结果
        outs = self.net.forward(self.output_layers)
        
        # 解析输出结果，寻找红绿灯
        for out in outs:
            for detection in out:
                # 获取所有类别的分数
                scores = detection[5:]
                # 获取最高分数的类别ID
                class_id = np.argmax(scores)
                # 获取置信度
                confidence = scores[class_id]
                
                # 如果是红绿灯且置信度大于阈值（降低阈值以提高检测率）
                if class_id == self.traffic_light_class_id and confidence > 0.3:
                    # 计算边界框坐标
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    
                    # 增加检测框大小，确保完全包裹整个红绿灯
                    w = int(detection[2] * width * 1.5)  # 增加宽度50%
                    h = int(detection[3] * height * 1.5)  # 增加高度50%
                    
                    # 计算左上角坐标
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    # 确保坐标在图像范围内
                    x = max(0, x)
                    y = max(0, y)
                    w = min(width - x, w)
                    h = min(height - y, h)
                    
                    # 添加到红绿灯框列表
                    traffic_light_boxes.append((x, y, w, h))
                    # 绘制蓝色矩形框标记红绿灯位置
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 如果检测到红绿灯，分析其颜色状态
        if len(traffic_light_boxes) > 0:
            # 对每个检测到的红绿灯进行颜色分析
            for (x, y, w, h) in traffic_light_boxes:
                # 提取红绿灯区域
                roi = frame[y:y+h, x:x+w]
                
                # 转换到HSV颜色空间
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # 定义红色和绿色的HSV范围（降低阈值以提高检测率）
                # 红色有两个范围
                lower_red1 = np.array([0, 80, 80])
                upper_red1 = np.array([15, 255, 255])
                lower_red2 = np.array([150, 80, 80])
                upper_red2 = np.array([180, 255, 255])
                
                lower_green = np.array([30, 80, 80])
                upper_green = np.array([80, 255, 255])
                
                # 创建红色和绿色的掩码
                mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask_red = mask_red1 + mask_red2
                mask_green = cv2.inRange(hsv, lower_green, upper_green)
                
                # 计算红色和绿色像素的数量
                red_pixels = cv2.countNonZero(mask_red)
                green_pixels = cv2.countNonZero(mask_green)
                
                # 设置最小像素数量阈值（降低阈值以提高检测率）
                min_pixels = max(5, roi.size * 0.005)  # 至少5个像素或占区域的0.5%
                
                # 判断红绿灯状态
                if red_pixels > green_pixels and red_pixels > min_pixels:
                    traffic_light_state = "红灯"
                    # 在红绿灯区域内绘制红色标记
                    frame = self.draw_chinese_text(frame, "红灯", (x, y - 20), font_size=16, color=(0, 0, 255))
                elif green_pixels > red_pixels and green_pixels > min_pixels:
                    traffic_light_state = "绿灯"
                    # 在红绿灯区域内绘制绿色标记
                    frame = self.draw_chinese_text(frame, "绿灯", (x, y - 20), font_size=16, color=(0, 255, 0))
                else:
                    # 不确定状态，可能是黄灯或未检测到明确颜色
                    frame = self.draw_chinese_text(frame, "未知", (x, y - 20), font_size=16, color=(255, 255, 255))
        else:
            # 没有检测到红绿灯，直接显示未知状态
            frame = self.draw_chinese_text(frame, "红绿灯: 未检测到", (10, 30), font_size=20, color=(255, 255, 255))
        
        # 显示整体红绿灯状态
        if traffic_light_state == "红灯":
            # 确保红灯状态显示为红色（RGB格式为255,0,0）
            frame = self.draw_chinese_text(frame, "红绿灯: 红灯", (10, 30), font_size=20, color=(255, 0, 0))
        elif traffic_light_state == "绿灯":
            # 确保绿灯状态显示为绿色（RGB格式为0,255,0）
            frame = self.draw_chinese_text(frame, "红绿灯: 绿灯", (10, 30), font_size=20, color=(0, 255, 0))
        else:
            frame = self.draw_chinese_text(frame, "红绿灯: 未知状态", (10, 30), font_size=20, color=(255, 255, 255))
            
        return frame, traffic_light_state
        

    def detect_jaywalking(self, frame, pedestrian_boxes, traffic_light_state):
        # 简单的闯红灯检测逻辑：
        # 如果检测到红灯，且在画面下半部分有行人，则判定为闯红灯
        jaywalking_boxes = []
        
        if traffic_light_state == "红灯" and pedestrian_boxes:
            # 定义道路区域（假设在画面下半部分）
            road_y_threshold = frame.shape[0] * 0.5
            
            for (x, y, w, h) in pedestrian_boxes:
                # 检查行人是否在道路区域内
                if y + h > road_y_threshold:
                    jaywalking_boxes.append((x, y, w, h))
                    # 用红色框标记闯红灯的行人
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    frame = self.draw_chinese_text(frame, "闯红灯", (x, y - 10), font_size=16, color=(0, 0, 255))
        
        # 显示闯红灯人数
        if jaywalking_boxes:
            # 红灯状态时显示红色
            text_color = (255, 0, 0) if traffic_light_state == "红灯" else (0, 0, 255)
            frame = self.draw_chinese_text(frame, f"闯红灯人数: {len(jaywalking_boxes)}", (10, 60), font_size=20, color=text_color)
        
        return frame, jaywalking_boxes
    
    def update_frame(self):
        """更新视频帧"""
        if self.camera is None or not self.is_running or self.is_paused:
            return
        
        ret, frame = self.camera.read()
        if not ret:
            # 如果是视频文件播放结束，重新开始播放
            if self.video_path is not None:
                self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.camera.read()
                if not ret:
                    self.stop_video()
                    QMessageBox.information(self, '提示', '视频播放完毕')
                    return
            else:
                self.stop_video()
                QMessageBox.information(self, '提示', '摄像头断开')
                return
        
        # 如果开启了红绿灯检测，先检测红绿灯再检测行人，避免行人框遮挡影响红绿灯识别
        traffic_light_state = "未知"
        jaywalking_boxes = []
        
        if self.traffic_light_detection_enabled:
            frame, traffic_light_state = self.detect_traffic_light(frame)
        
        # 检测行人
        frame, pedestrian_boxes = self.detect_pedestrians(frame)
        
        # 显示检测到的行人数量
        frame = self.draw_chinese_text(frame, f"行人数量: {len(pedestrian_boxes)}", (10, 90), font_size=20, color=(255, 255, 255))
        
        # 如果开启了红绿灯检测，再检测闯红灯行为
        if self.traffic_light_detection_enabled:
            frame, jaywalking_boxes = self.detect_jaywalking(frame, pedestrian_boxes, traffic_light_state)
        
        # 将OpenCV的BGR格式转换为Qt的RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 调整图像大小以适应标签
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # 在标签上显示图像
        self.video_label.setPixmap(scaled_pixmap)
    
    def toggle_pause(self):
        """切换视频播放的暂停/继续状态"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_button.setText('继续')
            self.status_label.setText('已暂停')
        else:
            self.pause_button.setText('暂停')
            # 恢复状态显示
            if self.video_path:
                self.status_label.setText(f'视频播放中: {os.path.basename(self.video_path)} - 正在进行行人识别...')
            else:
                self.status_label.setText('摄像头运行中...')
    
    def resizeEvent(self, event):
        # 窗口大小改变时，更新视频显示
        if hasattr(self.video_label, 'pixmap') and self.video_label.pixmap() is not None:
            scaled_pixmap = self.video_label.pixmap().scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)
    
    def closeEvent(self, event):
        # 关闭窗口时释放资源
        self.stop_video()
        event.accept()
    
    def draw_chinese_text(self, frame, text, position, font_size=20, color=(255, 255, 255)):
        """在OpenCV图像上绘制中文文本
        
        Args:
            frame: OpenCV图像 (BGR格式)
            text: 要绘制的中文文本
            position: 文本位置 (x, y)
            font_size: 字体大小
            color: 文本颜色 (RGB格式)
            
        Returns:
            绘制了文本的图像
        """
        # 将BGR转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img)
        
        # 尝试加载多种中文字体，增加成功率
        font = None
        fonts_to_try = [
            "simhei.ttf",          # 黑体
            "msyh.ttc",           # 微软雅黑
            "simsun.ttc",         # 宋体
            os.path.join(self.base_dir, "fonts", "simhei.ttf"),  # 程序目录下的字体
            os.path.join(os.path.dirname(os.path.abspath(sys.executable)), "fonts", "simhei.ttf")  # exe目录下的字体
        ]
        
        # 尝试在系统字体目录中查找
        if sys.platform == 'win32':
            fonts_to_try.extend([
                os.path.join("C:", "Windows", "Fonts", "simhei.ttf"),
                os.path.join("C:", "Windows", "Fonts", "msyh.ttc"),
                os.path.join("C:", "Windows", "Fonts", "simsun.ttc")
            ])
        
        # 尝试加载字体
        for font_path in fonts_to_try:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        # 如果所有尝试都失败，使用默认字体
        if font is None:
            font = ImageFont.load_default()
        
        # 绘制文本
        draw.text(position, text, font=font, fill=color)
        
        # 转换回OpenCV格式
        result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return result_img

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 设置中文字体
    font = app.font()
    font.setFamily("SimHei")
    app.setFont(font)
    
    window = PedestrianRecognitionApp()
    window.show()
    sys.exit(app.exec_())