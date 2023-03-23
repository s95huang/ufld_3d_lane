cmake_minimum_required(VERSION 3.10.0)
project(lane_detection_msgs)

set(CMAKE_CXX_STANDARD 17)


set(MESSAGE_FILES
  "LaneDetectionMsg.msg"
  "CNNLaneDetectionMsg.msg"
  "LaneDetectionMsgFull.msg"
)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
  sensor_msgs
)

add_message_files(FILES ${MESSAGE_FILES})

generate_messages(
  DEPENDENCIES
  std_msgs 
  sensor_msgs
)
