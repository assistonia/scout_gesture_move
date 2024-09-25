import cv2
import mediapipe as mp
import numpy as np
import torch
from cnn_lstm import CNN_LSTM
import base64
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from robot_controller import move_robot
from geometry_msgs.msg import Twist
import time

app = Flask(__name__, static_url_path='/static')
socketio = SocketIO(app, cors_allowed_origins="*")

# 모델 전체를 불러옵니다.
CNN_LSTM_model = torch.load('./CNN_LSTM_model_epoch_13_val_loss_0.0192.pt', map_location=torch.device('cpu'))

# 모델을 평가 모드로 설정합니다.
CNN_LSTM_model.eval()

actions = ['Select Drone', 'Select Group', 'Select Mode', 'ARM', 'DISARM', 'TAKEOFF', 'LAND', 'RTL', 'Change Altitude', 
           'Change Speed', 'Move Up', 'Move Down', 'Rotate CW', 'Move Forward', 'Move Backward', 'Move Right', 'Move Left', 
           'Rotate CCW', 'Cancel', 'Check', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten']

seq_length = 30

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

cap = cv2.VideoCapture(4)

seq = []
this_action = '?'
action_seq = [this_action]*seq_length

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/zed/zed_node/rgb/image_rect_color',  # 실제 사용 중인 ROS 토픽으로 변경해주세요
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.latest_image = None
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.is_moving = False
        self.last_action_time = time.time()
        self.action_cooldown = 2.0

    def move_robot_for_duration(self, x, y, z, duration):
        start_time = time.time()
        while time.time() - start_time < duration:
            move_robot(self, x, y, z)
            time.sleep(0.1)
        move_robot(self, 0, 0, 0)
        self.is_moving = False
        print("이동이 완료되었습니다. 다음 명령을 기다리는 중...")

    def listener_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # ROS 이미지를 받았을 때 웹 클라이언트로 전송
        if self.latest_image is not None:
            _, buffer = cv2.imencode('.jpg', self.latest_image)
            ros_img_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('ros_image', {'ros_image': ros_img_base64})

def process_frame():
    global seq, this_action, action_seq, pending_action

    ret, img = cap.read()
    if not ret:
        return None, None

    img = cv2.flip(img, 1)
    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)

    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 

            angle = np.degrees(angle)

            gesture_joint = np.concatenate([joint.flatten(), angle])
            seq.append(gesture_joint)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            input_data = torch.FloatTensor(input_data)

            y_pred = CNN_LSTM_model(input_data)
            _, indices = torch.max(y_pred, dim=1)

            action = actions[indices.item()]
            action_seq.append(action)

            if len(action_seq) < 3:
                continue

            this_action = '?'
            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action_seq[-1]

                current_time = time.time()
                if current_time - image_subscriber.last_action_time > image_subscriber.action_cooldown and not image_subscriber.is_moving:
                    image_subscriber.last_action_time = current_time
                    print(f"제스처 인식: {this_action}")  # 제스처 인식 로그 추가
                    if this_action in ['Move Forward', 'Move Backward', 'Move Right', 'Move Left']:
                        pending_action = this_action
                        print(f"명령 대기 중: {pending_action}")  # 명령 대기 로그 추가
                    elif this_action == 'Check' and pending_action:
                        print(f"명령 실행: {pending_action}")  # 명령 실행 로그 추가
                        if pending_action == 'Move Forward':
                            threading.Thread(target=image_subscriber.move_robot_for_duration, args=(0.5, 0, 0, 2.0)).start()
                        elif pending_action == 'Move Backward':
                            threading.Thread(target=image_subscriber.move_robot_for_duration, args=(-0.5, 0, 0, 2.0)).start()
                        elif pending_action == 'Move Right':
                            threading.Thread(target=image_subscriber.move_robot_for_duration, args=(0, 0, -1.57, 3.0)).start()
                            #time.sleep(5)
                            #threading.Thread(target=image_subscriber.move_robot_for_duration, args=(0.5, 0, 0, 2.0)).start()
                        elif pending_action == 'Move Left':
                            threading.Thread(target=image_subscriber.move_robot_for_duration, args=(0, 0, 1.57, 3.0)).start()
                            #time.sleep(5)
                            #threading.Thread(target=image_subscriber.move_robot_for_duration, args=(0.5, 0, 0, 2.0)).start()
                        pending_action = None
                    elif this_action == 'Cancel':
                        print("명령 취소")  # 명령 취소 로그 추가
                        pending_action = None
                        this_action = '?'
                    else:
                        print(f"이동 명령이 없는 제스처: {this_action}, 스킵합니다.")  # 이동 명령이 없는 제스처 로그 추가
                        image_subscriber.is_moving = False
                        this_action = '?'
                        continue

    # cv2.putText(img, f'Action: {this_action.upper()}', (10, 30), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return img, this_action


def generate_frames():
    while True:
        img, action = process_frame()
        if img is None:
            continue

        #  웹캠 이미지를 base64로 인코딩
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 소켓을 통해 이미지와 제스처 정보 전송
        if action not in ['Move Forward', 'Move Backward', 'Move Right', 'Move Left' , 'Cancel', 'Check']:
            action = '?'
        socketio.emit('image_update', {
            'webcam_image': img_base64,
            'gesture': action
        })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

def flask_thread():
    socketio.run(app, host='0.0.0.0', port=5000)

def ros_thread(node):
    rclpy.spin(node)

if __name__ == '__main__':
    rclpy.init()
    image_subscriber = ImageSubscriber()

    # Flask 서버를 별도의 스레드에서 실행
    threading.Thread(target=flask_thread, daemon=True).start()

    # ROS2 노드를 별도의 스레드에서 실행
    threading.Thread(target=ros_thread, args=(image_subscriber,), daemon=True).start()

    # 메인 루프 실행
    generate_frames()

    cv2.destroyAllWindows()
    cap.release()
    rclpy.shutdown()
