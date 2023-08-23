# -*- encoding: utf-8 -*-
#-------------------------------------------------#
# Date created          : 2020. 8. 18.
# Date last modified    : 2020. 8. 19.
# Author                : chamadams@gmail.com
# Site                  : http://wandlab.com
# License               : GNU General Public License(GPL) 2.0
# Version               : 0.1.0
# Python Version        : 3.6+
#-------------------------------------------------#

import time
import cv2
import imutils
import platform
import numpy as np
from threading import Thread
from queue import Queue


class Streamer:

    def __init__(self):

        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
        print('[wandlab] ', 'OpenCL : ', cv2.ocl.haveOpenCL())

        self.capture = None
        self.thread = None
        self.width = 640
        self.height = 360
        self.stat = False
        self.current_time = time.time()
        self.preview_time = time.time()
        self.sec = 0
        self.Q = Queue(maxsize=128)
        self.started = False

    def run(self, src=0):

        self.stop()

        if platform.system() == 'Windows':
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)

        else:
            self.capture = cv2.VideoCapture(src)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.thread is None:
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = False
            self.thread.start()

        self.started = True

    def stop(self):

        self.started = False

        if self.capture is not None:
            self.capture.release()
            self.clear()

    def update(self):

        while True:

            if self.started:
                (grabbed, frame) = self.capture.read()

                if grabbed:
                    self.Q.put(frame)

    def clear(self):

        with self.Q.mutex:
            self.Q.queue.clear()

    def read(self):

        return self.Q.get()

    def blank(self):

        return np.ones(shape=[self.height, self.width, 3], dtype=np.uint8)

    def bytescode(self):

        if not self.capture.isOpened():

            frame = self.blank()

        else:

            frame = imutils.resize(self.read(), width=int(self.width))

            if self.stat:
                cv2.rectangle(frame, (0, 0), (120, 30), (0, 0, 0), -1)
                fps = 'FPS : ' + str(self.fps())
                cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        return cv2.imencode('.jpg', frame)[1].tobytes()

    def fps(self):

        self.current_time = time.time()
        self.sec = self.current_time - self.preview_time
        self.preview_time = self.current_time

        if self.sec > 0:
            fps = round(1 / (self.sec), 1)

        else:
            fps = 1

        return fps

    def __exit__(self):
        print('* streamer class exit')
        self.capture.release()

import torch
import sys, os
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class Yolo5Streamer:

    def __init__(self,
        weights = './runs/train/x_100e_8b/best.pt',  # model path or triton URL
        source = 0,  # file/dir/URL/glob/screen/0(webcam)
        data= './data/DSmart.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        updateInterval = 60, # 몇 프레임에 한 번씩 db 업데이트 할 건지
        stockBuffSize = 200 # 버퍼에 몇 프레임 넣어놓고 수량 빈도 집계할 건지
        ):

        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
        print('[wandlab] ', 'OpenCL : ', cv2.ocl.haveOpenCL())

        self.capture = None
        self.thread = None
        self.stat = False
        self.width = 640
        self.height = 480
        self.current_time = time.time()
        self.preview_time = time.time()
        self.sec = 0
        self.Q = Queue(maxsize=128)
        self.started = False

        #
        self.view_img = view_img
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        self.classes = classes
        self.agnostic_nms = agnostic_nms

        self.augment = augment
        self.line_thickness = line_thickness

        #
        self.item_stock_dict = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 }
        self.item_id_dict = { 'jin': 1, 'sand': 2, 'kan': 3, 'chap': 4, 'fried': 5, 'shrimp': 6 }
        self.updateInterval = updateInterval
        self.stockBuffSize = stockBuffSize
        self.stockBuff = []
        self.before_stock = ""

        #
        check_requirements(exclude=('tensorboard', 'thop'))

        # Load model
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data, fp16=half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        # Run inference
        self.model.warmup(imgsz=(1, 3, *imgsz))  # warmup
        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())

    def run(self, src=0):

        self.stop()

        if platform.system() == 'Windows':
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)

        else:
            self.capture = cv2.VideoCapture(src)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.thread is None:
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()

        self.started = True

    def stop(self):

        self.started = False

        if self.capture is not None:
            self.capture.release()
            self.clear()

    def update(self):

        while True:
            # print("YOLO5Streamer.update(): thread process")
            if self.started:
                (grabbed, frame) = self.capture.read()

                if grabbed:
                    if self.Q.full():
                        self.clear()
                    self.Q.put(frame)
                    self.getModelInference(frame)

    def getModelInference(self, im):
        s = ''
        im0 = im.copy()

        im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]  # resize
        im = im[::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im[np.newaxis])  # contiguous

        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            pred = self.model(im, augment=self.augment)

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            self.seen += 1

            s += f'{i}: '
            s += '%gx%g ' % im.shape[2:]  # print string

            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                items = [0, 0, 0, 0, 0, 0]  # 매 프레임 탐지된 객체 갯수를 기록할 리스트
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    items[int(c)] = int(n)  # 해당 프레임에 탐지된 객체 갯수 기록

                self.stockBuff.append(items)
                if (len(self.stockBuff) > self.stockBuffSize):
                    self.stockBuff.pop(0)
                stockBuffNum = np.array(self.stockBuff).T
                for idx, item in enumerate(stockBuffNum):
                    frequency = np.bincount(item)
                    stock = frequency.argmax()
                    self.item_stock_dict[self.item_id_dict[self.names[idx]]] = stock

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            if self.view_img:
                cv2.imshow("DS Mart Detection View", im0)
                cv2.waitKey(1)  # 1 millisecond


        # Print time (inference-only)
        print(f"{s}{'' if len(det) else '(no detections), '}{self.dt[1].dt * 1E3:.1f}ms")
        del s, im0, im, pred, det

        if (self.seen % self.updateInterval == 0):
            if (self.before_stock != str(self.item_stock_dict)[1:-1]):
                print(self.item_stock_dict)
                data = {
                    'item_stock_list': str(self.item_stock_dict)[1:-1]  # 중괄호 제거한 딕셔너리 문자열 표현
                }
                """
                try:
                    response = requests.post(parent_url, data=data)
                    if (response.status_code != 200):
                        LOGGER.info("failed to send data")
                    else:
                        LOGGER.info("successed to send data")
                        before_stock = str(item_stock_dict)[1:-1]
                except:
                    LOGGER.info("Cannot connect to Server!")
                """

        # Print results
        t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)

    def clear(self):

        with self.Q.mutex:
            self.Q.queue.clear()

    def read(self):

        return self.Q.get()

    def blank(self):

        return np.ones(shape=[self.height, self.width, 3], dtype=np.uint8)

    def bytescode(self):

        if not self.capture.isOpened():

            frame = self.blank()

        else:

            frame = imutils.resize(self.read(), width=int(self.width))

            if self.stat:
                cv2.rectangle(frame, (0, 0), (120, 30), (0, 0, 0), -1)
                fps = 'FPS : ' + str(self.fps())
                cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        return cv2.imencode('.jpg', frame)[1].tobytes()

    def fps(self):

        self.current_time = time.time()
        self.sec = self.current_time - self.preview_time
        self.preview_time = self.current_time

        if self.sec > 0:
            fps = round(1 / (self.sec), 1)

        else:
            fps = 1

        return fps

    def __exit__(self):
        print('* streamer class exit')
        self.capture.release()