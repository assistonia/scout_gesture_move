# Source ROS 2 Humble setup
source /opt/ros/humble/setup.bash

# Set ROS 2 domain ID
export ROS_DOMAIN_ID=6

# Set the RMW (ROS Middleware) implementation to Fast DDS
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Set the Fast DDS profiles file location
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/pyongjoo/Desktop/fastdds_rpi.xml

export CYCLONEDDS_URI='<CycloneDDS><Domain><General><DontRoute>true</></></></>'

python3 /home/pyongjoo/Desktop/lastgesture/main_homepage_scout.py
