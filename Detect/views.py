from django.shortcuts import render
from django.http import HttpResponse

# Import các thư viện và module cần thiết
from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from .util import get_car, read_license_plate

def detect_vehicles(request):
    # Đặt mã code của bạn tại đây
    new_width = 800
    new_height = 600

    results = {}
    mot_tracker = Sort()

    # Load models
    coco_model = YOLO('./yolov8n.pt')
    license_plate_detector = YOLO('./models/best.pt')

    # Load video
    cap = cv2.VideoCapture('./videos/IMG_4424.mp4')

    vehicles = [2, 3, 5, 7]

    frame_nmr = -1
    ret = True
    frame_skip = 2  # Số frame bạn muốn bỏ qua giữa các lần xử lý

    # Danh sách biển số xe đã xử lý
    processed_license_plates = []

    while ret:
        frame_nmr += 1
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (new_width, new_height))
            results[frame_nmr] = {}
            if frame_nmr % frame_skip == 0:  # Bỏ qua frame không cần xử lý
                detections = coco_model(frame, classes=[2, 3, 5, 7])[0]
                detections_ = []

                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score])

                # Vẽ bounding boxes xung quanh các đối tượng đã phát hiện
                for box in detections_:
                    x1, y1, x2, y2, _ = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

            # Hiển thị video trong chế độ full màn hình
            cv2.imshow('Detected Video', frame)

            # Đợi một khoảng thời gian ngắn (vd: 10 ms) và kiểm tra nút bấm "q" để thoát
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # Track vehicles (nếu bạn muốn thực hiện tracking)
            if frame_nmr % frame_skip == 0 and detections_:
                track_ids = mot_tracker.update(np.asarray(detections_))

            # Detection và xử lý biển số
            if frame_nmr % frame_skip == 0:
                license_plates = license_plate_detector(frame)[0]
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    # Kiểm tra xem biển số đã được xử lý chưa
                    license_plate_text = None
                    if license_plate not in processed_license_plates:
                        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                        # crop license plate, progress, và đọc biển số
                        # (đoạn code xử lý biển số có thể cần tối ưu hóa để giảm lag)
                        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 90, 255, cv2.THRESH_BINARY_INV)

                        # Đọc biển số xe
                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                        cv2.imshow('origin_crop', license_plate_crop)
                        cv2.imshow('threshold_crop', license_plate_crop_thresh)

                        if license_plate_text is not None:
                            print('License Plate:', license_plate_text)
                            print('License Plate Score:', license_plate_text_score)

                        # Thêm biển số xe đã xử lý vào danh sách
                        processed_license_plates.append(license_plate)

            # Giải phóng tài nguyên và đóng cửa sổ OpenCV
    cap.release()
    cv2.destroyAllWindows()

    return HttpResponse("Xử lý hoàn thành!")  # Thay bằng phản hồi hoặc mẫu giao diện của bạn

