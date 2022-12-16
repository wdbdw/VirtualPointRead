import os
import sys
import numpy as np
from os import getcwd
import cv2
import msvcrt
from ctypes import *
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import time
import math
import pyttsx3
from baidu_pp_wrap import Baidu_PP_OCR
from picture_model import pic_cla 
from threading import Thread
from pic_lang_demo import Picture_langues
sys.path.append("D:\MVS\Development\Samples\Python\MvImport")
from MvCameraControl_class import *
g_bExit = False   
# 枚举设备
def enum_devices(device = 0 , device_way = False):
 
    if device_way == False:
        if device == 0:
            tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE | MV_UNKNOW_DEVICE | MV_1394_DEVICE | MV_CAMERALINK_DEVICE
            deviceList = MV_CC_DEVICE_INFO_LIST()
            # 枚举设备
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
            if ret != 0:
                print("enum devices fail! ret[0x%x]" % ret)
                sys.exit()
            if deviceList.nDeviceNum == 0:
                print("find no device!")
                sys.exit()
            print("Find %d devices!" % deviceList.nDeviceNum)
            return deviceList
        else:
            pass
    elif device_way == True:
        pass
class Draw_initialize:
    def __init__(self):

        self.hand_num = 0
        self.vi_ha_color = 'None'
        self.pic_flag=False
        self.last_finger_cord_x_flag = {'Left': 0, 'Right': 0}
        self.last_finger_cord_y_flag= {'Left': 0, 'Right': 0}
        self.last_finger_arc_degree = {'Left': 0, 'Right': 0}
        self.right_ha_cir_list = []
        now = time.time()
        self.stop_time = {'Left': now, 'Right': now}
        self.hand_ring_color = {'Left': (255, 180, 0), 'Right': (255, 160, 255)}
        self.pic_text = {'dog': '狗', 'cat': '猫','chair':'椅子','Cola':'无','scissors':'无','person':'无','bicycle':'自行车','car':'轿车','bus':'公共汽车','pizza':'披萨','apple':'苹果','orange':'橘子','bird':'鸟','book':'书','phone':'手机','hot dog':'热狗','bottle':'瓶子'}
        self.float_distance = 10
        self.activate_duration = 0.3
        self.single_dete_duration = 1
        self.single_dete_last_time = None
        self.last_th_pic = None
        self.text_f=''
        self.pic_rec=pic_cla()
        self.pp_ocr = Baidu_PP_OCR()
        self.picture_language=Picture_langues()
        self.engine=pyttsx3.init()
        self.rate=self.engine.getProperty('rate')
        self.engine.setProperty('rate',150)
        volume=self.engine.getProperty('volume')
        self.engine.setProperty('volume',1)
        self.last_obj_identify = {'dec':None,'ocr':'无'}
    def say(self,text):
        self.engine.say(text)
        self.engine.runAndWait()
        self.engine.stop()
    def cv_add_chincese_text(self,img, text, position, textColor=(0, 255, 0), textSize=30):
        if (isinstance(img, np.ndarray)): 
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
     
        draw = ImageDraw.Draw(img)
       
        fontStyle = ImageFont.truetype(
            "./fonts/simsun.ttc", textSize, encoding="utf-8")
        draw.text(position, text, textColor, font=fontStyle)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    def generateOcrTextArea(self,ocr_text,line_text_num,line_num,x, y, w, h,frame):
        # First we crop the sub-rect from the image
        sub_img = frame[y:y+h, x:x+w]
        green_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0

        res = cv2.addWeighted(sub_img, 0.5, green_rect, 0.5, 1.0)
        for i in range(line_num):
            text = ocr_text[(i*line_text_num):(i+1)*line_text_num]
            res  = self.cv_add_chincese_text(res, text, (10,30*i+10), textColor=(255, 198, 255), textSize=18)
        return res
    def cv_add_lable_area(self,text,x, y, w, h,frame):
        sub_img = frame[y:y+h, x:x+w]
        green_rect = np.ones(sub_img.shape, dtype=np.uint8)   * 0
        res = cv2.addWeighted(sub_img, 0.5, green_rect, 0.5, 1.0)
        res  = self.cv_add_chincese_text(res, text, (10,10), textColor=(196, 255, 255), textSize=30)
        return res
    def cv_add_thumb(self,raw_img,frame):
        img = cv2.imread('D.jpg',1)
        frame=frame.copy()
        if  self.last_obj_identify['dec'] == None:
            resu = self.pic_rec.pic_model_rec(raw_img)
            print(resu)
            if len(resu)>0:
                label_en = resu
                label_zh = self.pic_text[label_en]
                if(resu=='Cola' or resu=='scissors'):
                    label_en='NO'
                    label_zh ='无'
                print(label_en)
                print(label_zh)
                name=label_zh+'  '+label_en
                self.last_obj_identify['dec'] = [label_zh,label_en]
                if self.last_obj_identify['dec'][0]!='无':
                    obj = Thread(target=self.say, args=(name,))
                    obj.start()
            else:
                self.last_obj_identify['dec'] = ['无','None']
        frame_height, frame_width, _ = frame.shape
        # 覆盖
        raw_img_h, raw_img_w, _ = raw_img.shape
        th_pic_size = 300
        th_pic_h_size = math.ceil( raw_img_h * th_pic_size / raw_img_w)
        th_pic = cv2.resize(raw_img, (th_pic_size, th_pic_h_size))
        rect_weight = 4
        # 在缩略图上画框框
        th_pic = cv2.rectangle(th_pic,(0,0),(th_pic_size,th_pic_h_size),(180, 139, 247),rect_weight)
        x, y, w, h = (frame_width - th_pic_size),th_pic_h_size,th_pic_size,50
        # Putting the image back to its position
        if  self.last_obj_identify['dec'] != ['无','None']:
            frame[y:y+h, x:x+w] = self.cv_add_lable_area('{label_zh} {label_en}'.format(label_zh=self.last_obj_identify['dec'][0],label_en=self.last_obj_identify['dec'][1]),x, y, w, h,frame)
        ocr_text = ''
        if self.last_obj_identify['ocr'] == '无':
            src_im,text_list = self.pp_ocr.ocr_image(raw_img)
            th_pic = cv2.resize(src_im, (th_pic_size, th_pic_h_size))
            if len(text_list) > 0 :
                ocr_text = ''.join(text_list)
                print(len(ocr_text))
                print(type(ocr_text[0]))
                if(ocr_text[0].isdigit()):
                    num0=int(ocr_text[0])
                    num2=int(ocr_text[2])
                    if(ocr_text[1]=="+"):
                        num=num0+num2
                        self.last_obj_identify['ocr']= ocr_text+str(num)
                        print(self.last_obj_identify['ocr'])
                        self.text_f=ocr_text[0]+'加'+ocr_text[2]+'等于'+str(num)
                        obj = Thread(target=self.say, args=(self.text_f,))
                        obj.start()
                        self.text_f=''
                    if(ocr_text[1]=="-"):
                        
                        num=num0-num2
                        self.last_obj_identify['ocr']= ocr_text+str(num)
                        self.text_f=ocr_text[0]+'减'+ocr_text[2]+'等于'+str(num)
                        obj = Thread(target=self.say, args=(self.text_f,))
                        obj.start()
                        self.text_f=''
                    if(ocr_text[1]=="*"):
                        num=num0*num2
                        self.last_obj_identify['ocr']= ocr_text+str(num)
                        self.text_f=ocr_text[0]+'乘'+ocr_text[2]+'等于'+str(num)
                        obj = Thread(target=self.say, args=(self.text_f,))
                        obj.start()
                        self.text_f=''
                    if(ocr_text[1]=="/"):
                        num=num0/num2
                        self.last_obj_identify['ocr']= ocr_text+str(num)
                        self.text_f=ocr_text[0]+'除'+ocr_text[2]+'等于'+str(num)
                        obj = Thread(target=self.say, args=(self.text_f,))
                        obj.start()
                        self.text_f=''
                else:
                    
                        self.last_obj_identify['ocr']= ocr_text
                        
            else:
                # 检测过，无结果
                self.last_obj_identify['ocr']= 'checked_no'
        else:
            ocr_text =  self.last_obj_identify['ocr']
            if self.last_obj_identify['ocr']!='checked_no':
                obj = Thread(target=self.say, args=(ocr_text,))
                obj.start()
                self.text_f=''

        if(self.last_obj_identify['ocr']=='checked_no'):
            pic_la_txt=self.picture_language.generate_caption(raw_img)
            self.last_obj_identify['ocr']=pic_la_txt
            print(pic_la_txt)
        frame[0:th_pic_h_size,(frame_width - th_pic_size):frame_width,:] = th_pic
        if ocr_text != '' and ocr_text != 'checked_no' :
            line_text_num = 15
            line_num = math.ceil(len(ocr_text) / line_text_num)
            y,h = (y+h+20),(32*line_num)
            frame[y:y+h, x:x+w] = self.generateOcrTextArea(ocr_text,line_text_num,line_num,x, y, w, h,frame)
        self.last_th_pic = th_pic
        return frame




    # 画圆环
    def draw_circle_ring(self, frame, point_x, point_y, arc_radius=80, end=360, color = (255, 0, 255),width=10):

        img = Image.fromarray(frame)
        shape = [(point_x-arc_radius, point_y-arc_radius),
                 (point_x+arc_radius, point_y+arc_radius)]
        img1 = ImageDraw.Draw(img)
        img1.arc(shape, start=0, end=end, fill=color, width=width)
        frame = np.asarray(img)

        return frame
    def clear_danshou(self):
        self.vi_ha_color = 'None'
        self.right_ha_cir_list = []
        self.last_finger_arc_degree = {'Left': 0, 'Right': 0}
        self.single_dete_last_time = None
    
    def single_vi_ha_color(self,x_distance,y_distance,double_hand_flag, finger_cord, frame, frame_copy):
        self.right_ha_cir_list.append( (finger_cord[0],finger_cord[1]) )
        for i in range(len(self.right_ha_cir_list)-1) :
         
            frame = cv2.line(frame,self.right_ha_cir_list[i],self.right_ha_cir_list[i+1],(255,0,0),5)
        max_x = max(self.right_ha_cir_list,key=lambda i : i[0])[0]
        min_x = min(self.right_ha_cir_list,key=lambda i : i[0])[0]
        max_y = max(self.right_ha_cir_list,key=lambda i : i[1])[1]
        min_y = min(self.right_ha_cir_list,key=lambda i : i[1])[1]
        frame = cv2.rectangle(frame,(min_x,min_y),(max_x,max_y),(0,255,0),2)
        frame = self.draw_circle_ring(
                    frame, finger_cord[0], finger_cord[1], arc_radius=50, end=360, color=self.hand_ring_color[double_hand_flag],width=15)
        if (x_distance <= self.float_distance) and (y_distance <= self.float_distance):
            if (time.time() - self.single_dete_last_time ) > self.single_dete_duration :
                if( (max_y - min_y) > 100) and( (max_x-min_x) > 100):
                    print('激活')
                    if not isinstance(self.last_th_pic, np.ndarray):
                            
                        self.last_obj_identify = {'dec':None,'ocr':'无'}
                        raw_img = frame_copy[min_y:max_y,min_x:max_x,]
                        frame = self.cv_add_thumb(raw_img,frame)
                    

        else:
        # 移动，重新计时
            self.single_dete_last_time = time.time() # 记录一下时间
        return frame
    def Finger_Index_Move_flag(self,double_hand_flag, finger_cord, frame,frame_copy):
        x_distance = abs(finger_cord[0] - self.last_finger_cord_x_flag[double_hand_flag])
        y_distance = abs(finger_cord[1] - self.last_finger_cord_y[double_hand_flag])
        if self.vi_ha_color == 'single':
            if self.hand_num == 2:
               self.clear_danshou() 
            elif double_hand_flag == 'Right':
                frame = self.single_vi_ha_color(x_distance,y_distance,double_hand_flag, finger_cord, frame , frame_copy)
        else:
            if (x_distance <= self.float_distance) and (y_distance <= self.float_distance):
                if(time.time() - self.stop_time[double_hand_flag]) > self.activate_duration:
                    arc_degree = 5 * ((time.time() - self.stop_time[double_hand_flag] - self.activate_duration) // 0.01)
                    if arc_degree <= 360:
                        frame = self.draw_circle_ring(
                            frame, finger_cord[0], finger_cord[1], arc_radius=50, end=arc_degree, color=self.hand_ring_color[double_hand_flag], width=15)
                    else:
                        frame = self.draw_circle_ring(
                            frame, finger_cord[0], finger_cord[1], arc_radius=50, end=360, color=self.hand_ring_color[double_hand_flag],width=15)  
                        self.last_finger_arc_degree[double_hand_flag] = 360
                        
                       
                        if (self.last_finger_arc_degree['Left'] >= 360) and (self.last_finger_arc_degree['Right'] >= 360):
                           
                            rect_l = (self.last_finger_cord_x_flag['Left'],self.last_finger_cord_y['Left'])
                            rect_r = (self.last_finger_cord_x_flag['Right'],self.last_finger_cord_y['Right'])
                            frame = cv2.rectangle(frame,rect_l,rect_r,(180,180,120),2)

                            if  self.last_obj_identify['dec']:
                                x, y, w, h = self.last_finger_cord_x_flag['Left'],(self.last_finger_cord_y['Left']-50),120,50
                                frame[y:y+h, x:x+w] = self.cv_add_lable_area('{label_zh}'.format(label_zh=self.last_obj_identify['dec'][0]),x, y, w, h,frame)
                            if self.vi_ha_color != 'double':
                                self.last_obj_identify = {'dec':None,'ocr':'无'}
                                raw_img = frame_copy[self.last_finger_cord_y['Left']:self.last_finger_cord_y['Right'],self.last_finger_cord_x_flag['Left']:self.last_finger_cord_x_flag['Right'],]
                                frame = self.cv_add_thumb(raw_img,frame)
                                  
                            self.vi_ha_color = 'double'
                        if (self.hand_num==1) and (self.last_finger_arc_degree['Right'] == 360):
                            self.vi_ha_color = 'single'
                            self.single_dete_last_time = time.time() 
                            self.right_ha_cir_list.append( (finger_cord[0],finger_cord[1]) )
            else:
                self.stop_time[double_hand_flag] = time.time()
                self.last_finger_arc_degree[double_hand_flag] = 0
        self.last_finger_cord_x_flag[double_hand_flag] = finger_cord[0]
        self.last_finger_cord_y[double_hand_flag] = finger_cord[1]
        return frame
class Reader_main:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.draw_operation = Draw_initialize()
        self.image=None
    def check_handa_index(self,double_hand_flag):
        if len(double_hand_flag) == 1:
            double_hand_flag_list = ['Left' if  double_hand_flag[0].classification[0].label == 'Right' else 'Right']
        else:
            double_hand_flag_list = [double_hand_flag[1].classification[0].label,double_hand_flag[0].classification[0].label]
        return double_hand_flag_list


   
    def re_main(self,img1):
        
        resize_w = 1024
        resize_h = 768
        with self.mp_hands.Hands(min_detection_confidence=0.7,
                                 min_tracking_confidence=0.5,
                                 max_num_hands=2) as hands:

            self.image=img1
            self.image.flags.writeable = False
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            results = hands.process(self.image)
            self.image.flags.writeable = True
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            if isinstance(self.draw_operation.last_th_pic, np.ndarray):
                self.image  = self.draw_operation.cv_add_thumb(self.draw_operation.last_th_pic,self.image )

            hand_num = 0
            if results.multi_hand_landmarks:
                double_hand_flag_list =  self.check_handa_index(results.multi_double_hand_flag)
                hand_num = len(double_hand_flag_list)
                self.draw_operation.hand_num = hand_num
                frame_copy = self.image.copy()
                for hand_index,hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if hand_index>1:
                        hand_index = 1  
                    self.mp_drawing.draw_landmarks(
                        self.image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())
                    landmark_list = []
                    paw_x_list = []
                    paw_y_list = []
                    for landmark_id, finger_axis in enumerate(
                            hand_landmarks.landmark):
                        landmark_list.append([
                            landmark_id, finger_axis.x, finger_axis.y,
                            finger_axis.z
                        ])
                        paw_x_list.append(finger_axis.x)
                        paw_y_list.append(finger_axis.y)
                    if landmark_list:
                        ratio_x_to_pixel = lambda x: math.ceil(x * resize_w)
                        ratio_y_to_pixel = lambda y: math.ceil(y * resize_h)
                        paw_left_top_x,paw_right_bottom_x = map(ratio_x_to_pixel,[min(paw_x_list),max(paw_x_list)])
                        paw_left_top_y,paw_right_bottom_y = map(ratio_y_to_pixel,[min(paw_y_list),max(paw_y_list)])
                        index_finger_tip = landmark_list[8]
                        index_finger_tip_x =ratio_x_to_pixel(index_finger_tip[1])
                        index_finger_tip_y =ratio_y_to_pixel(index_finger_tip[2])
                        middle_finger_tip = landmark_list[12]
                        middle_finger_tip_x =ratio_x_to_pixel(middle_finger_tip[1])
                        middle_finger_tip_y =ratio_y_to_pixel(middle_finger_tip[2])
                        l_r_hand_text = double_hand_flag_list[hand_index][:1]
                        cv2.putText(self.image, "{hand} x:{x} y:{y}".format(hand=l_r_hand_text,x=index_finger_tip_x,y=index_finger_tip_y) , (paw_left_top_x-30+10,paw_left_top_y-40),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 180, 255), 2)
                        cv2.rectangle(self.image,(paw_left_top_x-30,paw_left_top_y-30),(paw_right_bottom_x+30,paw_right_bottom_y+30),(180, 139, 247),1)
                        line_len = math.hypot((index_finger_tip_x-middle_finger_tip_x),(index_finger_tip_y-middle_finger_tip_y))
                        if line_len < 50 and double_hand_flag_list[hand_index] == 'Right':
                            self.draw_operation.clear_danshou()
                            self.draw_operation.last_th_pic = None
                        self.image = self.draw_operation.Finger_Index_Move_flag(double_hand_flag_list[hand_index],[index_finger_tip_x,index_finger_tip_y],self.image,frame_copy)
            self.image = self.draw_operation.cv_add_chincese_text(self.image, "虚拟点读学习系统" , (10, 30), textColor=(0,160,180), textSize=50)
            cv2.imshow('virtual', self.image)

            k = cv2.waitKey(27) & 0xff
control = Reader_main() 
def identify_different_devices(deviceList):
    # 判断不同类型设备，并输出相关信息
    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        # 判断是否为网口相机
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print ("\n网口设备序号: [%d]" % i)
            # 获取设备名
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print ("当前设备型号名: %s" % strModeName)
            # 获取当前设备 IP 地址
            nip1_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip1_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip1_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip1_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print ("当前 ip 地址: %d.%d.%d.%d" % (nip1_1, nip1_2, nip1_3, nip1_4))
            # 获取当前子网掩码
            nip2_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0xff000000) >> 24)
            nip2_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0x00ff0000) >> 16)
            nip2_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0x0000ff00) >> 8)
            nip2_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentSubNetMask & 0x000000ff)
            print ("当前子网掩码 : %d.%d.%d.%d" % (nip2_1, nip2_2, nip2_3, nip2_4))
            # 获取当前网关
            nip3_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0xff000000) >> 24)
            nip3_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0x00ff0000) >> 16)
            nip3_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0x0000ff00) >> 8)
            nip3_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nDefultGateWay & 0x000000ff)
            print("当前网关 : %d.%d.%d.%d" % (nip3_1, nip3_2, nip3_3, nip3_4))
            # 获取网口 IP 地址
            nip4_1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0xff000000) >> 24)
            nip4_2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0x00ff0000) >> 16)
            nip4_3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0x0000ff00) >> 8)
            nip4_4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nNetExport & 0x000000ff)
            print("当前连接的网口 IP 地址 : %d.%d.%d.%d" % (nip4_1, nip4_2, nip4_3, nip4_4))
            # 获取制造商名称
            strmanufacturerName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chManufacturerName:
                strmanufacturerName = strmanufacturerName + chr(per)
            print("制造商名称 : %s" % strmanufacturerName)
            # 获取设备版本
            stdeviceversion = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chDeviceVersion:
                stdeviceversion = stdeviceversion + chr(per)
            print("设备当前使用固件版本 : %s" % stdeviceversion)
            # 获取制造商的具体信息
            stManufacturerSpecificInfo = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chManufacturerSpecificInfo:
                stManufacturerSpecificInfo = stManufacturerSpecificInfo + chr(per)
            print("设备制造商的具体信息 : %s" % stManufacturerSpecificInfo)
            # 获取设备序列号
            stSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chSerialNumber:
                stSerialNumber = stSerialNumber + chr(per)
            print("设备序列号 : %s" % stSerialNumber)
            # 获取用户自定义名称
            stUserDefinedName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                stUserDefinedName = stUserDefinedName + chr(per)
            print("用户自定义名称 : %s" % stUserDefinedName)
 
        # 判断是否为 USB 接口相机
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print ("\nU3V 设备序号e: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print ("当前设备型号名 : %s" % strModeName)
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print ("当前设备序列号 : %s" % strSerialNumber)
            # 获取制造商名称
            strmanufacturerName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chVendorName:
                strmanufacturerName = strmanufacturerName + chr(per)
            print("制造商名称 : %s" % strmanufacturerName)
            # 获取设备版本
            stdeviceversion = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chDeviceVersion:
                stdeviceversion = stdeviceversion + chr(per)
            print("设备当前使用固件版本 : %s" % stdeviceversion)
            # 获取设备序列号
            stSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                stSerialNumber = stSerialNumber + chr(per)
            print("设备序列号 : %s" % stSerialNumber)
            # 获取用户自定义名称
            stUserDefinedName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chUserDefinedName:
                stUserDefinedName = stUserDefinedName + chr(per)
            print("用户自定义名称 : %s" % stUserDefinedName)
            # 获取设备 GUID
            stDeviceGUID = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chDeviceGUID:
                stDeviceGUID = stDeviceGUID + chr(per)
            print("设备GUID号 : %s" % stDeviceGUID)
            # 获取设备的家族名称
            stFamilyName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chFamilyName:
                stFamilyName = stFamilyName + chr(per)
            print("设备的家族名称 : %s" % stFamilyName)
 
        # 判断是否为 1394-a/b 设备
        elif mvcc_dev_info.nTLayerType == MV_1394_DEVICE:
            print("\n1394-a/b device: [%d]" % i)
 
        # 判断是否为 cameralink 设备
        elif mvcc_dev_info.nTLayerType == MV_CAMERALINK_DEVICE:
            print("\ncameralink device: [%d]" % i)
            # 获取当前设备名
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("当前设备型号名 : %s" % strModeName)
            # 获取当前设备序列号
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("当前设备序列号 : %s" % strSerialNumber)
            # 获取制造商名称
            strmanufacturerName = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chVendorName:
                strmanufacturerName = strmanufacturerName + chr(per)
            print("制造商名称 : %s" % strmanufacturerName)
            # 获取设备版本
            stdeviceversion = ""
            for per in mvcc_dev_info.SpecialInfo.stCamLInfo.chDeviceVersion:
                stdeviceversion = stdeviceversion + chr(per)
            print("设备当前使用固件版本 : %s" % stdeviceversion)
 
# 输入需要连接的相机的序号
def input_num_camera(deviceList):
    nConnectionNum = input("please input the number of the device to connect:")
    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("intput error!")
        sys.exit()
    return nConnectionNum
 
# 创建相机实例并创建句柄,(设置日志路径)
def creat_camera(deviceList , nConnectionNum ,log = True , log_path = getcwd()):
    """
    :param deviceList:        设备列表
    :param nConnectionNum:    需要连接的设备序号
    :param log:               是否创建日志
    :param log_path:          日志保存路径
    :return:                  相机实例和设备列表
    """
    # 创建相机实例
    cam = MvCamera()
    # 选择设备并创建句柄
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
    if log == True:
        ret = cam.MV_CC_SetSDKLogPath(log_path)
        print(log_path)
        if ret != 0:
            print("set Log path  fail! ret[0x%x]" % ret)
            sys.exit()
        # 创建句柄,生成日志
        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()
    elif log == False:
        # 创建句柄,不生成日志
        ret = cam.MV_CC_CreateHandleWithoutLog(stDeviceList)
        print(1111)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()
    return cam , stDeviceList
 
# 打开设备
def open_device(cam):
    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()
 
# 获取各种类型节点参数
def get_Value(cam , param_type = "int_value" , node_name = "PayloadSize"):
    """
    :param cam:            相机实例
    :param_type:           获取节点值得类型
    :param node_name:      节点名 可选 int 、float 、enum 、bool 、string 型节点
    :return:               节点值
    """
    if param_type == "int_value":
        stParam = MVCC_INTVALUE_EX()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
        ret = cam.MV_CC_GetIntValueEx(node_name, stParam)
        if ret != 0:
            print("获取 int 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name , ret))
            sys.exit()
        int_value = stParam.nCurValue
        return int_value
 
    elif param_type == "float_value":
        stFloatValue = MVCC_FLOATVALUE()
        memset(byref(stFloatValue), 0, sizeof(MVCC_FLOATVALUE))
        ret = cam.MV_CC_GetFloatValue( node_name , stFloatValue)
        if ret != 0:
            print("获取 float 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name , ret))
            sys.exit()
        float_value = stFloatValue.fCurValue
        return float_value
 
    elif param_type == "enum_value":
        stEnumValue = MVCC_ENUMVALUE()
        memset(byref(stEnumValue), 0, sizeof(MVCC_ENUMVALUE))
        ret = cam.MV_CC_GetEnumValue(node_name, stEnumValue)
        if ret != 0:
            print("获取 enum 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name , ret))
            sys.exit()
        enum_value = stEnumValue.nCurValue
        return enum_value
 
    elif param_type == "bool_value":
        stBool = c_bool(False)
        ret = cam.MV_CC_GetBoolValue(node_name, stBool)
        if ret != 0:
            print("获取 bool 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name , ret))
            sys.exit()
        return stBool.value
 
    elif param_type == "string_value":
        stStringValue =  MVCC_STRINGVALUE()
        memset(byref(stStringValue), 0, sizeof( MVCC_STRINGVALUE))
        ret = cam.MV_CC_GetStringValue(node_name, stStringValue)
        if ret != 0:
            print("获取 string 型数据 %s 失败 ! 报错码 ret[0x%x]" % (node_name , ret))
            sys.exit()
        string_value = stStringValue.chCurValue
        return string_value
 
# 设置各种类型节点参数
def set_Value(cam , param_type = "int_value" , node_name = "PayloadSize" , node_value = None):
    """
    :param cam:               相机实例
    :param param_type:        需要设置的节点值得类型
        int:
        float:
        enum:     参考于客户端中该选项的 Enum Entry Value 值即可
        bool:     对应 0 为关，1 为开
        string:   输入值为数字或者英文字符，不能为汉字
    :param node_name:         需要设置的节点名
    :param node_value:        设置给节点的值
    :return:
    """
    if param_type == "int_value":
        stParam = int(node_value)
        ret = cam.MV_CC_SetIntValueEx(node_name, stParam)
        if ret != 0:
            print("设置 int 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name , ret))
            sys.exit()
        print("设置 int 型数据节点 %s 成功 ！设置值为 %s !"%(node_name , node_value))
 
    elif param_type == "float_value":
        stFloatValue = float(node_value)
        ret = cam.MV_CC_SetFloatValue( node_name , stFloatValue)
        if ret != 0:
            print("设置 float 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name , ret))
            sys.exit()
        print("设置 float 型数据节点 %s 成功 ！设置值为 %s !" % (node_name, node_value))
 
    elif param_type == "enum_value":
        stEnumValue = node_value
        ret = cam.MV_CC_SetEnumValue(node_name, stEnumValue)
        if ret != 0:
            print("设置 enum 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name , ret))
            sys.exit()
        print("设置 enum 型数据节点 %s 成功 ！设置值为 %s !" % (node_name, node_value))
 
    elif param_type == "bool_value":
        ret = cam.MV_CC_SetBoolValue(node_name, node_value)
        if ret != 0:
            print("设置 bool 型数据节点 %s 失败 ！ 报错码 ret[0x%x]" %(node_name,ret))
            sys.exit()
        print("设置 bool 型数据节点 %s 成功 ！设置值为 %s !" % (node_name, node_value))
 
    elif param_type == "string_value":
        stStringValue = str(node_value)
        ret = cam.MV_CC_SetStringValue(node_name, stStringValue)
        if ret != 0:
            print("设置 string 型数据节点 %s 失败 ! 报错码 ret[0x%x]" % (node_name , ret))
            sys.exit()
        print("设置 string 型数据节点 %s 成功 ！设置值为 %s !" % (node_name, node_value))
 
# 寄存器读写
def read_or_write_memory(cam , way = "read"):
    if way == "read":
        pass
        cam.MV_CC_ReadMemory()
    elif way == "write":
        pass
        cam.MV_CC_WriteMemory()
 
# 判断相机是否处于连接状态(返回值如何获取)=================================
def decide_divice_on_line(cam):
    value = cam.MV_CC_IsDeviceConnected()
    if value == True:
        print("该设备在线 ！")
    else:
        print("该设备已掉线 ！", value)
 
# 设置 SDK 内部图像缓存节点个数
def set_image_Node_num(cam , Num = 1):
    ret = cam.MV_CC_SetImageNodeNum(nNum = Num)
    if ret != 0:
        print("设置 SDK 内部图像缓存节点个数失败 ,报错码 ret[0x%x]" % ret)
    else:
        print("设置 SDK 内部图像缓存节点个数为 %d  ，设置成功!" % Num)
 
# 设置取流策略
def set_grab_strategy(cam , grabstrategy = 0 , outputqueuesize = 1):
    """
    • OneByOne: 从旧到新一帧一帧的从输出缓存列表中获取图像，打开设备后默认为该策略
    • LatestImagesOnly: 仅从输出缓存列表中获取最新的一帧图像，同时清空输出缓存列表
    • LatestImages: 从输出缓存列表中获取最新的OutputQueueSize帧图像，其中OutputQueueSize范围为1 - ImageNodeNum，可用MV_CC_SetOutputQueueSize()接口设置，ImageNodeNum默认为1，可用MV_CC_SetImageNodeNum()接口设置OutputQueueSize设置成1等同于LatestImagesOnly策略，OutputQueueSize设置成ImageNodeNum等同于OneByOne策略
    • UpcomingImage: 在调用取流接口时忽略输出缓存列表中所有图像，并等待设备即将生成的一帧图像。该策略只支持GigE设备，不支持U3V设备
    """
    if grabstrategy != 2:
        ret = cam.MV_CC_SetGrabStrategy(enGrabStrategy = grabstrategy)
        if ret != 0:
            print("设置取流策略失败 ,报错码 ret[0x%x]" % ret)
        else:
            print("设置 取流策略为 %d  ，设置成功!" % grabstrategy)
    else:
        ret = cam.MV_CC_SetGrabStrategy(enGrabStrategy=grabstrategy)
        if ret != 0:
            print("设置取流策略失败 ,报错码 ret[0x%x]" % ret)
        else:
            print("设置 取流策略为 %d  ，设置成功!" % grabstrategy)
 
        ret = cam.MV_CC_SetOutputQueueSize(nOutputQueueSize = outputqueuesize)
        if ret != 0:
            print("设置使出缓存个数失败 ,报错码 ret[0x%x]" % ret)
        else:
            print("设置 输出缓存个数为 %d  ，设置成功!" % outputqueuesize)



# 显示图像
def image_show(image):
    image = cv2.resize(image, (1024, 768), interpolation=cv2.INTER_AREA)
    control.re_main(image)

# 需要显示的图像数据转换
def image_control(data , stFrameInfo):
    if stFrameInfo.enPixelType == 17301505:
        image = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
        image_show(image=image , name = stFrameInfo.nHeight)
    elif stFrameInfo.enPixelType == 17301514:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_GB2RGB)
        image_show(image=image, name = stFrameInfo.nHeight)
    elif stFrameInfo.enPixelType == 35127316:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        image_show(image=image, name = stFrameInfo.nHeight)
    elif stFrameInfo.enPixelType == PixelType_Gvsp_BGR8_Packed :
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_Y422)
        image_show(image = image, name = stFrameInfo.nHeight)
def IsImageColor(enType):
    dates = {
        PixelType_Gvsp_RGB8_Packed: 'color',
        PixelType_Gvsp_BGR8_Packed: 'color',
        PixelType_Gvsp_YUV422_Packed: 'color',
        PixelType_Gvsp_YUV422_YUYV_Packed: 'color',
        PixelType_Gvsp_BayerGR8: 'color',
        PixelType_Gvsp_BayerRG8: 'color',
        PixelType_Gvsp_BayerGB8: 'color',
        PixelType_Gvsp_BayerBG8: 'color',
        PixelType_Gvsp_BayerGB10: 'color',
        PixelType_Gvsp_BayerGB10_Packed: 'color',
        PixelType_Gvsp_BayerBG10: 'color',
        PixelType_Gvsp_BayerBG10_Packed: 'color',
        PixelType_Gvsp_BayerRG10: 'color',
        PixelType_Gvsp_BayerRG10_Packed: 'color',
        PixelType_Gvsp_BayerGR10: 'color',
        PixelType_Gvsp_BayerGR10_Packed: 'color',
        PixelType_Gvsp_BayerGB12: 'color',
        PixelType_Gvsp_BayerGB12_Packed: 'color',
        PixelType_Gvsp_BayerBG12: 'color',
        PixelType_Gvsp_BayerBG12_Packed: 'color',
        PixelType_Gvsp_BayerRG12: 'color',
        PixelType_Gvsp_BayerRG12_Packed: 'color',
        PixelType_Gvsp_BayerGR12: 'color',
        PixelType_Gvsp_BayerGR12_Packed: 'color',
        PixelType_Gvsp_Mono8: 'mono',
        PixelType_Gvsp_Mono10: 'mono',
        PixelType_Gvsp_Mono10_Packed: 'mono',
        PixelType_Gvsp_Mono12: 'mono',
        PixelType_Gvsp_Mono12_Packed: 'mono'}
    return dates.get(enType, '未知')
 
# 主动图像采集
def access_get_image(cam , active_way = "getImagebuffer"):
    """
    :param cam:     相机实例
    :active_way:主动取流方式的不同方法 分别是（getImagebuffer）（getoneframetimeout）
    :return:
    """
       #global img_buff
    img_buff = None
    stOutFrame = MV_FRAME_OUT()
    memset(byref(stOutFrame), 0, sizeof(stOutFrame))
    while True:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if None != stOutFrame.pBufAddr and 0 == ret:
            print ("MV_CC_GetImageBuffer: Width[%d], Height[%d], nFrameNum[%d]"  % (stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nFrameNum))
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            if IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'mono':
                print("mono!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                nConvertSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight
            elif IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'color':
                print("color!")
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed  # opecv要用BGR，不能使用RGB
                nConvertSize = stOutFrame.stFrameInfo.nWidth * stOutFrame.stFrameInfo.nHeight * 3
            else:
                print("not support!!!")
            if img_buff is None:
                img_buff = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
            stConvertParam.nWidth = stOutFrame.stFrameInfo.nWidth
            stConvertParam.nHeight = stOutFrame.stFrameInfo.nHeight
            stConvertParam.pSrcData = cast(stOutFrame.pBufAddr, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType
            stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
            stConvertParam.nDstBufferSize = nConvertSize
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                print("convert pixel fail! ret[0x%x]" % ret)
                del stConvertParam.pSrcData
                sys.exit()
            else:
                print("convert ok!!")
                # # 存raw图看看转化成功没有
                # file_path = "AfterConvert_RGB.raw"
                # file_open = open(file_path.encode('ascii'), 'wb+')
                # try:
                #     image_save= (c_ubyte * stConvertParam.nDstBufferSize)()
                #     cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                #     file_open.write(img_buff)
                #     print("raw ok!!")
                # except:
                #     raise Exception("save file executed failed:%s" % e.message)
                # finally:
                #     file_open.close()
            # 黑白处理
            if IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'mono':
                img_buff = (c_ubyte * stConvertParam.nDstLen)()
                cdll.msvcrt.memcpy(byref(img_buff),stConvertParam.pDstBuffer,stConvertParam.nDstLen)
                img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)
                img_buff = img_buff.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
                print("mono ok!!")
                image_show(image=img_buff)  # 显示图像函数
            # 彩色处理
            if IsImageColor(stOutFrame.stFrameInfo.enPixelType) == 'color':
                img_buff = (c_ubyte * stConvertParam.nDstLen)()
                cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
                img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstBufferSize), dtype=np.uint8)#data以流的形式读入转化成ndarray对象
                img_buff = img_buff.reshape(stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth,3)
                print("color ok!!")
                image_show(image=img_buff)  # 显示图像函数
            else:
                print("no data[0x%x]" % ret)   
        nRet = cam.MV_CC_FreeImageBuffer(stOutFrame)
        if g_bExit == True:
            break
winfun_ctype = WINFUNCTYPE
stFrameInfo = POINTER(MV_FRAME_OUT_INFO_EX)
pData = POINTER(c_ubyte)
FrameInfoCallBack = winfun_ctype(None, pData, stFrameInfo, c_void_p)
def image_callback(pData, pFrameInfo, pUser):
    global img_buff
    img_buff = None
    stFrameInfo = cast(pFrameInfo, POINTER(MV_FRAME_OUT_INFO_EX)).contents
    if stFrameInfo:
        print ("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
    if img_buff is None and stFrameInfo.enPixelType == 17301505:
        img_buff = (c_ubyte * stFrameInfo.nWidth*stFrameInfo.nHeight)()
        cdll.msvcrt.memcpy(byref(img_buff) , pData , stFrameInfo.nWidth*stFrameInfo.nHeight)
        data = np.frombuffer(img_buff , count = int(stFrameInfo.nWidth*stFrameInfo.nHeight) , dtype = np.uint8)
        image_control(data=data, stFrameInfo=stFrameInfo)
        del img_buff
    elif img_buff is None and stFrameInfo.enPixelType == 17301514:
        img_buff = (c_ubyte * stFrameInfo.nWidth*stFrameInfo.nHeight)()
        cdll.msvcrt.memcpy(byref(img_buff) , pData , stFrameInfo.nWidth*stFrameInfo.nHeight)
        data = np.frombuffer(img_buff , count = int(stFrameInfo.nWidth*stFrameInfo.nHeight) , dtype = np.uint8)
        image_control(data=data, stFrameInfo=stFrameInfo)
        del img_buff
    elif img_buff is None and stFrameInfo.enPixelType == 35127316:
        img_buff = (c_ubyte * stFrameInfo.nWidth * stFrameInfo.nHeight*3)()
        cdll.msvcrt.memcpy(byref(img_buff), pData, stFrameInfo.nWidth * stFrameInfo.nHeight*3)
        data = np.frombuffer(img_buff, count=int(stFrameInfo.nWidth * stFrameInfo.nHeight*3), dtype=np.uint8)
        image_control(data=data, stFrameInfo=stFrameInfo)
        del img_buff
    elif img_buff is None and stFrameInfo.enPixelType == 34603039:
        img_buff = (c_ubyte * stFrameInfo.nWidth * stFrameInfo.nHeight * 2)()
        cdll.msvcrt.memcpy(byref(img_buff), pData, stFrameInfo.nWidth * stFrameInfo.nHeight * 2)
        data = np.frombuffer(img_buff, count=int(stFrameInfo.nWidth * stFrameInfo.nHeight * 2), dtype=np.uint8)
        image_control(data=data, stFrameInfo=stFrameInfo)
        del img_buff
CALL_BACK_FUN = FrameInfoCallBack(image_callback)
 
# 事件回调
stEventInfo = POINTER(MV_EVENT_OUT_INFO)
pData = POINTER(c_ubyte)
EventInfoCallBack = winfun_ctype(None, stEventInfo, c_void_p)
def event_callback(pEventInfo, pUser):
    stPEventInfo = cast(pEventInfo, POINTER(MV_EVENT_OUT_INFO)).contents
    nBlockId = stPEventInfo.nBlockIdHigh
    nBlockId = (nBlockId << 32) + stPEventInfo.nBlockIdLow
    nTimestamp = stPEventInfo.nTimestampHigh
    nTimestamp = (nTimestamp << 32) + stPEventInfo.nTimestampLow
    if stPEventInfo:
        print ("EventName[%s], EventId[%u], BlockId[%d], Timestamp[%d]" % (stPEventInfo.EventName, stPEventInfo.nEventID, nBlockId, nTimestamp))
CALL_BACK_FUN_2 = EventInfoCallBack(event_callback)
 
# 注册回调取图
def call_back_get_image(cam):
    # ch:注册抓图回调 | en:Register image callback
    ret = cam.MV_CC_RegisterImageCallBackEx(CALL_BACK_FUN, None)
    if ret != 0:
        print("register image callback fail! ret[0x%x]" % ret)
        sys.exit()
 
# 关闭设备与销毁句柄
def close_and_destroy_device(cam , data_buf=None):
    # 停止取流
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        sys.exit()
    # 关闭设备
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()
    # 销毁句柄
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()
    del data_buf
 
# 开启取流并获取数据包大小
def start_grab_and_get_data_size(cam):
    ret = cam.MV_CC_StartGrabbing()
    k = cv2.waitKey(27) & 0xff
    if ret != 0:
        print("开始取流失败! ret[0x%x]" % ret)
        sys.exit()
 
def main():
    # 枚举设备
    deviceList = enum_devices(device=0, device_way=False)
    # 判断不同类型设备
    identify_different_devices(deviceList)
    # 输入需要被连接的设备
    nConnectionNum = input_num_camera(deviceList)
    # 创建相机实例并创建句柄,(设置日志路径)
    cam, stDeviceList = creat_camera(deviceList, nConnectionNum, log=False)
    # decide_divice_on_line(cam)  ==============
    # 打开设备
    open_device(cam)
    stdcall = input("回调方式取流显示请输入 0    主动取流方式显示请输入 1:")
    if int(stdcall) == 0:
        # 回调方式抓取图像
        call_back_get_image(cam)
        # 开启设备取流
        start_grab_and_get_data_size(cam)
        # 当使用 回调取流时，需要在此处添加
        print ("press a key to stop grabbing.")
        msvcrt.getch()
        # 关闭设备与销毁句柄
        close_and_destroy_device(cam)
    elif int(stdcall) == 1:
        # 开启设备取流
        start_grab_and_get_data_size(cam)
        # 主动取流方式抓取图像
        access_get_image(cam, active_way="getImagebuffer")
        
        # 关闭设备与销毁句柄
        close_and_destroy_device(cam)
if __name__=="__main__":
    main()