import cv2
class pic_cla:
    def pic_model_rec(self,img):
        net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=(320, 320), scale=1/255)
        classes = []
        with open("dnn_model/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                classes.append(class_name)

        (class_ids, scores, bboxes) = model.detect(img, confThreshold=0.3, nmsThreshold=.4)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = classes[class_id]

        return(class_name)
