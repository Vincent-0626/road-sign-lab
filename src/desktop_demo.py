"""Local interactive interface built on the upstream GTSRB classifier."""
from __future__ import annotations

import sys
import threading
import queue
import ctypes
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageOps, ImageTk

from .config import CHECKPOINT_DIR, PROJECT_ROOT
from .predict import load_for_inference
from .robustness_demo import ZH_CLASSES, predict_array, transform_image
from .utils import configure_threads

if sys.platform == 'win32':
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


class Demo(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('交通标志识别 · 场景干扰实验')
        self.geometry('1040x720')
        self.minsize(920, 640)
        self.configure(bg='#f3f6fa')
        self.model = None
        self.image = None
        self.preview = None
        self.modified_preview = None
        self.status = tk.StringVar(value='正在加载模型，请稍候…')
        self.brightness = tk.DoubleVar(value=1.0)
        self.blur = tk.StringVar(value='无')
        self.occlusion = tk.BooleanVar(value=False)
        self.load_queue = queue.Queue()
        self._layout()
        threading.Thread(target=self._load, daemon=True).start()
        self.after(100, self._poll_load)

    def _layout(self):
        tk.Label(self, text='交通标志识别', font=('Microsoft YaHei UI', 23, 'bold'),
                 bg='#f3f6fa', fg='#18314f').pack(anchor='w', padx=30, pady=(18, 1))
        tk.Label(self, text='选择一张标志图片，观察亮度、模糊和遮挡对分类结果的影响',
                 font=('Microsoft YaHei UI', 10), bg='#f3f6fa', fg='#596b7b').pack(anchor='w', padx=32)
        container = tk.Frame(self, bg='#f3f6fa')
        container.pack(fill='both', expand=True, padx=30, pady=16)
        left = tk.Frame(container, bg='white', padx=18, pady=14)
        left.pack(side='left', fill='both', expand=True)
        right = tk.Frame(container, bg='white', padx=18, pady=14, width=340)
        right.pack(side='right', fill='y', padx=(12, 0))
        right.pack_propagate(False)
        self.image_label = tk.Label(left, text='请选择交通标志图片', bg='#edf2f7', fg='#748395',
                                    font=('Microsoft YaHei UI', 12), height=14)
        self.image_label.pack(fill='both', expand=True)
        self.filename = tk.StringVar(value='使用内置示例或选择本地图片')
        tk.Label(left, textvariable=self.filename, bg='white', fg='#586d80',
                 font=('Microsoft YaHei UI', 9)).pack(anchor='w', pady=(8, 4))
        buttons = tk.Frame(left, bg='white')
        buttons.pack(fill='x')
        ttk.Button(buttons, text='选择图片', command=self.choose).pack(side='left', padx=(0, 8))
        ttk.Button(buttons, text='停车让行示例', command=lambda: self.open_image(PROJECT_ROOT / 'samples/sample_stop.ppm')).pack(side='left', padx=(0, 8))
        ttk.Button(buttons, text='限速 30 示例', command=lambda: self.open_image(PROJECT_ROOT / 'samples/sample_speed_30.ppm')).pack(side='left')
        tk.Label(right, text='场景干扰', bg='white', fg='#18314f', font=('Microsoft YaHei UI', 15, 'bold')).pack(anchor='w')
        tk.Label(right, text='亮度', bg='white', fg='#526577').pack(anchor='w', pady=(12, 0))
        tk.Scale(right, from_=0.2, to=1.8, resolution=0.1, orient='horizontal', variable=self.brightness,
                 bg='white', highlightthickness=0, length=285).pack(anchor='w')
        tk.Label(right, text='模糊程度', bg='white', fg='#526577').pack(anchor='w', pady=(8, 2))
        ttk.Combobox(right, textvariable=self.blur, values=['无', '轻度', '中度', '重度'], state='readonly').pack(fill='x')
        tk.Checkbutton(right, text='模拟部分遮挡', variable=self.occlusion, bg='white',
                       activebackground='white').pack(anchor='w', pady=(8, 8))
        self.run_button = ttk.Button(right, text='分析图片', command=self.run, state='disabled')
        self.run_button.pack(fill='x', pady=(2, 10))
        tk.Label(right, text='候选结果', bg='white', fg='#18314f', font=('Microsoft YaHei UI', 14, 'bold')).pack(anchor='w', pady=(4, 8))
        self.result = tk.StringVar(value='选择图片后点击“分析图片”')
        tk.Label(right, textvariable=self.result, justify='left', anchor='nw', bg='white',
                 fg='#243c52', font=('Microsoft YaHei UI', 11), wraplength=285).pack(fill='x')
        tk.Label(right, text='分数为模型输出，不代表实际道路安全概率。', bg='white', fg='#7a8997',
                 font=('Microsoft YaHei UI', 9), wraplength=285, justify='left').pack(side='bottom', anchor='w')
        tk.Label(self, textvariable=self.status, bg='#f3f6fa', fg='#526577',
                 font=('Microsoft YaHei UI', 9)).pack(anchor='w', padx=32, pady=(0, 12))

    def _load(self):
        try:
            configure_threads()
            loaded = load_for_inference(CHECKPOINT_DIR / 'best.pt')
            self.load_queue.put(('ok', loaded))
        except Exception as error:
            self.load_queue.put(('error', str(error)))

    def _poll_load(self):
        try:
            status, value = self.load_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_load)
            return
        if status == 'ok':
            self._loaded(value)
        else:
            self.status.set('模型加载失败：' + value)

    def _loaded(self, loaded):
        self.model, self.cfg, _, self.norm, self.device = loaded
        self.status.set('模型已就绪 · 请使用裁剪好的交通标志图片')
        self.open_image(PROJECT_ROOT / 'samples/sample_stop.ppm')

    def choose(self):
        path = filedialog.askopenfilename(filetypes=[('图片', '*.jpg *.jpeg *.png *.bmp *.ppm *.webp'), ('所有文件', '*.*')])
        if path:
            self.open_image(Path(path))

    def open_image(self, path):
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((1600, 1600))
                self.image = np.asarray(image).copy()
            self.filename.set('当前图片：' + path.name)
            self.result.set('点击“分析图片”查看候选结果')
            self._show(self.image)
            if self.model is not None:
                self.run_button.configure(state='normal')
        except (OSError, ValueError) as error:
            self.status.set('无法读取图片：' + str(error))

    def _show(self, image):
        preview = Image.fromarray(image)
        scale = min(460 / preview.width, 360 / preview.height)
        preview = preview.resize((max(1, round(preview.width * scale)),
                                  max(1, round(preview.height * scale))), Image.Resampling.BICUBIC)
        self.preview = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.preview, text='', height=0)

    def run(self):
        if self.model is None or self.image is None:
            return
        try:
            blur = {'无': 0, '轻度': 3, '中度': 5, '重度': 7}[self.blur.get()]
            altered = transform_image(self.image, self.brightness.get(), blur, 0.3 if self.occlusion.get() else 0.0)
            before = predict_array(self.model, self.cfg, self.norm, self.device, self.image)
            after = predict_array(self.model, self.cfg, self.norm, self.device, altered)
            self._show(altered)
            lines = [f'原图：{ZH_CLASSES[before[0][0]]}  {before[0][1]:.1%}', '', '当前图片：']
            lines.extend(f'{i}. {ZH_CLASSES[class_id]}  {score:.1%}' for i, (class_id, score) in enumerate(after, 1))
            self.result.set('\n'.join(lines))
            self.status.set('分析完成 · 模型处理整张图片，请尽量裁剪到单个标志')
        except Exception as error:
            self.status.set('分析失败：' + str(error))


def main():
    try:
        Demo().mainloop()
    except tk.TclError as error:
        print(f'无法启动图形界面：{error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
