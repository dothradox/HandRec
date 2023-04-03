import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'ka', 1: 'kha', 2: 'ga', 3:'4',4:'5'}
nepali_alphabet = {
    0: 'ka',
    1: 'kha',
    2: 'ga',
    3: 'gha',
    4: 'nga',
    5: 'cha',
    6: 'chha',
    7: 'ja',
    8: 'jha',
    9: 'yna',
    10: 'ta',
    11: 'tha',
    12: 'da',
    13: 'dha',
    14: 'na',
    15: 'taa',
    16: 'thaa',
    17: 'daa',
    18: 'dhaa',
    19: 'naa',
    20: 'pa',
    21: 'pha',
    22: 'ba',
    23: 'bha',
    24: 'ma',
    25: 'ya',
    26: 'ra',
    27: 'la',
    28: 'wa',
    29: 'sha',
    30: 'shaa',
    31: 'sa',
    32: 'ha',
    33: 'ksha',
    34: 'tra',
    35: 'gya',
    36: 'shra'
}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

            # for i in range(len(hand_landmarks.landmark)):
            #     x = hand_landmarks.landmark[i].x
            #     y = hand_landmarks.landmark[i].y
            #     data_aux.append(x - min(x_))
            #     data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = nepali_alphabet[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

