import cv2
import mediapipe as mp
import numpy as np
import torch
from cnn_lstm import CNN_LSTM

# 모델 전체를 불러옵니다.
CNN_LSTM_model = torch.load('./CNN_LSTM_model_epoch_13_val_loss_0.0192.pt', map_location=torch.device('cpu'))

# 모델을 평가 모드로 설정합니다.
CNN_LSTM_model.eval()

gesture = {'Select Drone':0, 'Select Group':1, 'Select Mode':2, 'ARM':3, 'DISARM':4, 'TAKEOFF':5, 'LAND':6, 'RTL':7,
           'Change Altitude':8, 'Change Speed':9, 'Move Up':10, 'Move Down':11, 'Rotate CW':12, 'Move Forward':13, 
           'Move Backward':14, 'Move Right':15, 'Move Left':16, 'Rotate CCW':17, 'Cancel':18, 'Check':19, 
           'One':20, 'Two':21, 'Three':22, 'Four':23, 'Five':24, 'Six':25, 'Seven':26, 'Eight':27, 'Nine':28, 'Ten':29}

actions = ['Select Drone', 'Select Group', 'Select Mode', 'ARM', 'DISARM', 'TAKEOFF', 'LAND', 'RTL', 'Change Altitude', 
           'Change Speed', 'Move Up', 'Move Down', 'Rotate CW', 'Move Forward', 'Move Backward', 'Move Right', 'Move Left', 
           'Rotate CCW', 'Cancel', 'Check', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']

seq_length = 30


# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils         #landmark 그려주는 함수 패키지
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

cap = cv2.VideoCapture(4)

seq = []
# left_hand_action_seq = []
# right_hand_action_seq = []

# Unable to recognize the gesture if the last 3 gestures are different
this_action = '?'
action_seq = [this_action]*seq_length
number = 0
while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)

    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # idx = 0
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21,4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
            v = v2 - v1 

            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            angle = np.degrees(angle) 

            gesture_joint = np.array([angle], dtype=np.float32)
            gesture_joint = np.concatenate([joint.flatten(), angle])

            seq.append(gesture_joint)

            mp_drawing.draw_landmarks(
                        img, 
                        res, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(), 
                        mp_drawing_styles.get_default_hand_connections_style())

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            input_data=torch.FloatTensor(input_data)

            y_pred = CNN_LSTM_model(input_data)
            values, indices = torch.max(y_pred.data, dim=1,keepdim=True)

            conf = values

            # If the confidence is less than 0.9, the gesture is not recognized
            if conf < 0.98:
                continue 

            action = actions[indices]
            action_seq.append(action) 

            # if len(action_seq) < 3:
            #     continue
            if len(action_seq) < 10:
                continue

            this_action = '?'
        number +=1
        
        # If the last 3 gestures are the same, the gesture is recognized correctly
        if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4] == action_seq[-5] == action_seq[-6] == action_seq[-7] == action_seq[-8] == action_seq[-9] == action_seq[-10]:
            this_action = action_seq[-1]
            # print(this_action)
        print(this_action)

    text_position = (150, img.shape[0] - 100)
    # cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.putText(img, f'{this_action.upper()}', org=text_position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    cv2.imshow('Result of Gesture', img)
    cv2.imwrite('../240414/result_'+ str(number) + '.jpg', img)
   

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()