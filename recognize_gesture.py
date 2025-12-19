import cv2
import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model
from cnn_tf import CNNModel
import sqlite3
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ===================== LOAD TRAINED MODEL =====================

model = load_model(
    "cnn_model_keras2.keras",
    custom_objects={"CNNModel": CNNModel}
)

# ===================== IMAGE SIZE =====================

def get_image_size():
    img = cv2.imread("gestures/0/100.jpg", 0)
    return img.shape

image_x, image_y = get_image_size()

# ===================== IMAGE PROCESSING =====================

def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = img.astype(np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_probab = model.predict(processed, verbose=0)[0]
    pred_class = np.argmax(pred_probab)
    return max(pred_probab), pred_class

# ===================== DATABASE =====================

def get_pred_text_from_db(pred_class):
    conn = sqlite3.connect("gesture_db.db")
    cursor = conn.execute(
        "SELECT g_name FROM gesture WHERE g_id=" + str(pred_class)
    )
    for row in cursor:
        return row[0]
    return ""   # ✅ NEVER return None

# ===================== TEXT UTILS =====================

def split_sentence(text, num_of_words):
    if text is None or text.strip() == "":
        return []

    words = text.split(" ")
    result = []
    for i in range(0, len(words), num_of_words):
        result.append(" ".join(words[i:i + num_of_words]))
    return result

def put_splitted_text_in_blackboard(blackboard, splitted_text):
    y = 200
    for line in splitted_text:
        cv2.putText(
            blackboard,
            line,
            (20, y),
            cv2.FONT_HERSHEY_TRIPLEX,
            1.8,
            (255, 255, 255),
            2
        )
        y += 50

# ===================== HAND HISTOGRAM =====================

def get_hand_hist():
    with open("hist", "rb") as f:
        return pickle.load(f)

# ===================== MAIN RECOGNITION =====================

def recognize():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("❌ Camera not found")
        return

    hist = get_hand_hist()
    x, y, w, h = 300, 100, 300, 300

    while True:
        text = ""

        ret, img = cam.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))

        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)

        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)

        thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y:y+h, x:x+w]

        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        if contours:
            contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(contour) > 10000:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                save_img = thresh[y1:y1+h1, x1:x1+w1]

                if w1 > h1:
                    pad = (w1 - h1) // 2
                    save_img = cv2.copyMakeBorder(
                        save_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, 0
                    )
                elif h1 > w1:
                    pad = (h1 - w1) // 2
                    save_img = cv2.copyMakeBorder(
                        save_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, 0
                    )

                pred_probab, pred_class = keras_predict(model, save_img)

                if pred_probab * 100 > 80:
                    text = get_pred_text_from_db(pred_class)
                    print("Recognized:", text)

        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        splitted_text = split_sentence(text, 2)
        put_splitted_text_in_blackboard(blackboard, splitted_text)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        result = np.hstack((img, blackboard))

        cv2.imshow("Gesture Recognition", result)
        cv2.imshow("Threshold", thresh)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

# ===================== RUN =====================

keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
recognize()
