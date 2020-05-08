import cv2 as cv
import numpy as np
from time import time
from math import sqrt, floor

# Функция выводит название только необходимы нам слоев
def getOutputsNames(net):
    # Выводим названия всех слоёв в сетке
    layersNames = net.getLayerNames()

    # Выводим названия только слоев с несоединенными выводами (?)
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#------------------------------------------------------------------------------------------------------------

# Функция рисует боксы для масок на кадре
def yolo_postprocess(frame, outs):
    # Размеры кадра
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []

    confidences = []

    boxes = []

    # out - массив выходных данных из одного слоя ОДНОГО кадра(всего слоев несколько)
    for out in outs:
        # detection - это один распознанный на этом слое объект
        for detection in out:

            # извлекаем ID класса и вероятность
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Если "уверенность" больше минимального значения, то находим координаты боксы
            if confidence > mask_threshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)

                # Обновим все три ранее созданных массива
                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])


    # сейчас имеем заполненные массивы боксов и ID для одного кадра
    # применим non-maxima suppression чтобы отфильтровать накладывающиеся ненужные боксы
    # для КАЖДОГО кадра indices обновляются
    indices = cv.dnn.NMSBoxes(boxes, confidences, mask_threshold, nms_threshold)

    for i in indices:
        # То есть мы "отфильтровали" накладывающиеся боксы и сейчас СНОВА получаем координаты уже
        # отфильтрованных боксов
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        width = box[2] # Это именно ШИРИНА,а не координата x правого нижнего угла
        height = box[3]
        mask_box_coords = [x, y, width, height]


        # Название класса
        label = '%.2f' % confidences[i]
        # Получаем название класса и "уверенность"
        if classes:
            assert (classIDs[i]< len(classes))
            label = '%s:%s' % (classes[classIDs[i]], label)


        # Рисуем бокс и название класса
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(y, labelSize[1])
        cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv.putText(frame, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return mask_box_coords


#------------------------------------------------------------------------------------------------------------

# Функция рисует боксы для лиц на кадре и заполняет внешний массив координат лиц
# Также проверяется, надета ли маска
def vino_face_postprocess(frame, outs, mask_box_coords):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    for detection in outs.reshape(-1, 7):
        confidence = float(detection[2])
        x = int(detection[3] * frame.shape[1])
        y = int(detection[4] * frame.shape[0])
        width = int(detection[5] * frame.shape[1]) - x   # Это именно ширина, а не координтата нижнего правого угла
        height = int(detection[6] * frame.shape[0]) - y

        # Получаем координаты лица
        face_box_coords = [x, y, x + width, y + height]

        if confidence > face_threshold:
            cropped_face = frame[y:height, x:width]
            # Координаты лица добавляются в массив
            cropped_faces.append(cropped_face)
            # Название класса
            label = '%.2f' % confidence
            label = '%s:%s' % ("face", label)

            # Изначально будем считать, что маска не надета
            mask_inside = False
            status = "No mask"
            status_color = (0, 0, 255)

            # Проверяем, находится ли центр маски внутри бокса лица и меняем цвет бокса маски
            if (face_box_coords != []) and (mask_box_coords != []) and (face_box_coords is not None) and (mask_box_coords is not None):
                mask_inside = check_if_mask_inside_face(face_box_coords, mask_box_coords)


            # Если маска надета, то лицо рисуется зеленым, иначе - красным
            if mask_inside:
                color = (0, 255, 0)
                status = "Mask is on"
                status_color = (0, 255, 0)
            else:
                color = (0, 0, 255)

            # Статус маски пишется в правом нижнем углу
            cv.putText(frame, status, (int(frame_width / 2 ) - 40, frame_height-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

            # Рисуем бокс лица и название
            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y = max(y, labelSize[1])
            cv.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv.putText(frame, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            return face_box_coords


# Функция сравнивает два найденных бокса человека
def vino_person_compare(data, box):
    vino_person_re_blob = cv.dnn.blobFromImage(box, size=vino_person_re_net_size, ddepth=cv.CV_8U)
    vino_person_re_net.setInput(vino_person_re_blob)
    vino_person_re_outs = vino_person_re_net.forward()
    vino_person_re_outs = vino_person_re_outs.reshape(256)
    vino_person_re_outs /= sqrt(np.dot(vino_person_re_outs, vino_person_re_outs))
    ide = 1
    distance = -1

    if len(data) != 0:
        for x in data:
            distance = np.dot(vino_person_re_outs, data[x])
            ide += 1
            if distance > distance_threshold:
                ide = x
                break

    if distance < distance_threshold:
        data['id{}'.format(ide)] = vino_person_re_outs

    return distance, ide

# Функция рисует боксы для человека, а также отслеживает человека
def vino_person_postprocess(frame, outs):
    data = {}
    objects = 0
    for detection in outs.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])

        # Получаем ID человека

        if confidence > person_threshold:
            objects += 1
            box = frame[ymin:ymax, xmin:xmax]

            # Рисуется первый бокс человека
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 215, 0), 2)

            # Пробуем сравнить два бокса человека, чтобы отследить его передвижение
            try:
                distance, ID = vino_person_compare(data, box)
            except:
                continue

            # На кадре рисуется ID человека
            cv.putText(frame, 'person {}'.format(ID), (xmin, ymax - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Возвращаем координаты бокса человека
    person_box_coords = [xmin, ymin, xmax, ymax]
    return person_box_coords

#------------------------------------------------------------------------------------------------------------
# Функция проверяет, находится ли центр бокса маски в пределах бокса лица
def check_if_mask_inside_face(face_box_coords, mask_box_coords):

    face_x = face_box_coords[0]
    face_y = face_box_coords[1]
    face_width = face_box_coords[2]
    face_height = face_box_coords[3]

    mask_x = mask_box_coords[0]
    mask_y = mask_box_coords[1]
    mask_width = mask_box_coords[2]
    mask_height = mask_box_coords[3]

    # Получаем координаты середины бокса маски
    mask_center_x = int(floor(mask_x + (mask_x + mask_width)) / 2)
    mask_center_y = int(floor(mask_y + (mask_y + mask_height)) / 2)

    # Это массивы координат для проверки
    face_hor = range(face_x, face_x + face_width)
    face_vert = range(face_y, face_y + face_height)

    # Если координты центра маски есть в обоих массивах, то маска надета
    if (mask_center_x in face_hor) and (mask_center_y in face_vert):
        return True

    return False


# ------------------------------------------------------------------------------------------------------------



        ######### ОСНОВНОЕ ТЕЛО ПРОГРАММЫ #########

# Объявляем некоторые полезные переменные
# Минимальная вероятность для маски - 10 процентов
mask_threshold = 0.5
nms_threshold = 0.1

# Минимальная вероятность для лица - 90 процентов
face_threshold = 0.9

# Минимальная вероятность для человека - 90 процентов
person_threshold = 0.9

# Минимальное расстояние между боксами при отслеживании человека
distance_threshold = 0.4

# Количество кадров, через которое будет производится распознавание маски
step = 1

##################################################3

# Размеры входного изображения
inpWidth = 608
inpHeight = 608

# Файл с названиями классов
classesFile = "classes.names"

# Считываем названия классов
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Массив с координатами лиц на видео
cropped_faces = []
#_________________________________________________________________________________________
# Файл конфига для yolov3
yolo_conf = "yolov3.cfg"
# Файл с весами для yolov3
yolo_weights = 'yolo-obj_final.weights'
#_________________________________________________________________________________________
# Файлы конфига для vino_face
vino_face_xml = "face-detection-retail-0005/FP32/face-detection-retail-0005.xml"
vino_face_bin = "face-detection-retail-0005/FP32/face-detection-retail-0005.bin"

# Файлы конфига для vino_person_detection
vino_person_detection_xml = "person-detection-retail-0013/FP32/person-detection-retail-0013.xml"
vino_person_detection_bin = "person-detection-retail-0013/FP32/person-detection-retail-0013.bin"

# Файлы конфига для vino_person_reidentification
vino_person_re_xml = "person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.xml"
vino_person_re_bin = "person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.bin"
#_________________________________________________________________________________________

# YOLOv3
# Считываются все данные для yolo
yolo_net = cv.dnn.readNetFromDarknet(yolo_conf, yolo_weights)

# Она настраивается
yolo_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
#_________________________________________________________________________________________
# OpenVINO
# Считываются все данные для openvino
# Сетка для распознавания лица
vino_face_net = cv.dnn.readNet(vino_face_xml, vino_face_bin)
# Сетка для распознавания человека
vino_person_detection_net = cv.dnn.readNet(vino_person_detection_xml, vino_person_detection_bin)
# Сетка для подтверждения человека
vino_person_re_net = cv.dnn.readNet(vino_person_re_xml, vino_person_re_bin)
# Настраиваем openvino
vino_face_net_size = (300, 300)
vino_person_detection_net_size = (544, 320)
vino_person_re_net_size = (64, 160)

vino_face_net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
vino_face_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

vino_person_detection_net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
vino_person_detection_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

vino_person_re_net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)
vino_person_re_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


#_________________________________________________________________________________________


# Создатеся окно с названием "frame"
winName = 'frame'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
cv.resizeWindow(winName, 608, 608)


#cap = cv.VideoCapture("test_imgs/doctor.mp4")
cap = cv.VideoCapture(0)
grab, frame = cap.read()

# Счетчик кадров
count = 0
while True:
    # Запускаем отсчёт времени работы
    start = time()
    grab, frame = cap.read()
    if not grab:
        print("Video Not Found!!!")
        break
    count += 1
    resized = cv.resize(frame, (608,608), interpolation = cv.INTER_AREA)
    # Создаем каплю(?) из кадра и передаем ее в сеть для анализа
    # Это производится НЕ каждый кадр для ускорения работы
    if count % step == 0:

        yolo_blob = cv.dnn.blobFromImage(resized, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        vino_face_blob = cv.dnn.blobFromImage(frame, size=vino_face_net_size, ddepth=cv.CV_8U)
        vino_person_detection_blob = cv.dnn.blobFromImage(frame, size=vino_person_detection_net_size, ddepth=cv.CV_8U)

        # Помещаем данные в ТРИ сетки: лицо, маска, человек
        yolo_net.setInput(yolo_blob)
        vino_face_net.setInput(vino_face_blob)
        vino_person_detection_net.setInput(vino_person_detection_blob)

        # Получаем выходные данные c yolo
        yolo_outs = yolo_net.forward(getOutputsNames(yolo_net))

        # Получаем выходные данные c vino_face
        vino_face_outs = vino_face_net.forward()

        # Получаем выходные данные c vino_person
        vino_person_outs = vino_person_detection_net.forward()

        # Данные с yolo обрабатываются в функции
        mask_box_coords = yolo_postprocess(resized, yolo_outs)

        # Данные с vino_face обрабатываются в функции
        face_box_coords = vino_face_postprocess(resized, vino_face_outs, mask_box_coords)

        # Данные с vino_person обрабатываются в функции
        person_box_coords = vino_person_postprocess(resized, vino_person_outs)



    # Завершаем отсчёт времени работы для вычисления FPS
    end = time()
    fps = 1 / (end - start)
    cv.putText(resized, 'fps:{:.2f}'.format(fps + 3), (5, 25),cv.FONT_HERSHEY_SIMPLEX, 1, (255, 144, 30), 2)

    # Кадр со всеми нарисованными боксами показывается
    cv.imshow(winName, resized)

    # При нажатии на "q" все окна закрываются
    if cv.waitKey(14) & 0xFF == ord('q'):
        break

# Все окна закрываются
cv.destroyAllWindows()

