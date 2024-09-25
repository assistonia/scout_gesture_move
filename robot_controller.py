from geometry_msgs.msg import Twist

def move_robot(node, x, y, z):
    """
    로봇의 속도와 방향을 설정하는 함수
    :param node: 이미 생성된 노드
    :param x: linear.x 값 (전진 속도)
    :param y: linear.y 값 (옆으로 이동 속도)
    :param z: angular.z 값 (회전 속도)
    """
    # Twist 메시지 생성
    vel_msg = Twist()
    vel_msg.linear.x = float(x)
    vel_msg.linear.y = float(y)
    vel_msg.angular.z = float(z)

    # 퍼블리셔 생성 및 메시지 퍼블리시
    cmd_vel_pub = node.create_publisher(Twist, '/cmd_vel', 10)
    cmd_vel_pub.publish(vel_msg)

    # 로그 출력
    node.get_logger().info(f'로봇이 linear.x = {x}, linear.y = {y}, angular.z = {z} 로 이동 중입니다.')
