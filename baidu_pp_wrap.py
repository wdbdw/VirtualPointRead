
import sys
import os
import cv2
sys.path.append("baidu_pp_detection\\python")
import numpy as np



sys.path.append("baidu_pp_ocr\\tools\\infer")
sys.path.append("baidu_pp_ocr\\")
import baidu_pp_ocr.tools.infer.utility  as utility
from baidu_pp_ocr.tools.infer.predict_system import TextSystem
from baidu_pp_ocr.ppocr.utils.logging import get_logger
logger = get_logger()
# OCR
class Baidu_PP_OCR:
    def __init__(self):

        args = utility.parse_args()
        args.det_model_dir="./baidu_pp_ocr/models/ch_PP-OCRv2_det_infer/"
        args.rec_model_dir="./baidu_pp_ocr/models/ch_PP-OCRv2_rec_infer/"
        args.rec_char_dict_path="./baidu_pp_ocr/ppocr/utils/ppocr_keys_v1.txt"
        args.use_angle_cls=False 
        args.use_gpu=True
        self.text_sys = TextSystem(args)
        if 1:
            print('Warm up ocr model')
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = self.text_sys(img)
    def ocr_image(self,img):
        dt_boxes, rec_res = self.text_sys(img)
        text_list = []
        for text, score in rec_res:
                # logger.info("{}, {:.3f}".format(text, score))
                text_list.append(text)
        src_im = img
        for box in dt_boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        return src_im,text_list
