cmake_minimum_required(VERSION 3.8)
project(rm_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++17)
endif()

# ==========================================
# 1. 寻找基础依赖包
# ==========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# 【新增】寻找自定义消息所需的基础包
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ==========================================
# 2. 生成自定义消息 (必须放在编译节点之前)
# ==========================================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Armor.msg"
  "msg/Armors.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

# 【现代 ROS2 核心修复】提取生成的消息 C++ 支持库，存入变量 cpp_typesupport_target
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ==========================================
# 3. 配置 ONNXRuntime 核心引擎库
# ==========================================
set(ORT_DIR "/home/causin/onnxruntime_cpp")
include_directories(include ${ORT_DIR}/include)
link_directories(${ORT_DIR}/lib)

set(CMAKE_BUILD_RPATH "${ORT_DIR}/lib")
set(CMAKE_INSTALL_RPATH "${ORT_DIR}/lib")

# ==========================================
# 4. 编译核心算法库 (纯 C++ 库)
# ==========================================
add_library(yolo_detector SHARED src/yolo_detector.cpp)
target_link_libraries(yolo_detector ${OpenCV_LIBRARIES} onnxruntime)
target_include_directories(yolo_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

add_library(traditional_detector SHARED src/traditional_detector.cpp)
target_link_libraries(traditional_detector ${OpenCV_LIBRARIES})
target_include_directories(traditional_detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# ==========================================
# 5. 编译本地调试程序 (Standalone Test)
# ==========================================
add_executable(test_yolo_local src/test_yolo_local.cpp)
target_link_libraries(test_yolo_local yolo_detector ${OpenCV_LIBRARIES})

add_executable(test_traditional_local src/test_traditional_local.cpp)
target_link_libraries(test_traditional_local traditional_detector ${OpenCV_LIBRARIES})

# ==========================================
# 6. 编译 ROS 2 节点
# ==========================================
add_executable(image_source_node src/Image_Source_Node.cpp)
ament_target_dependencies(image_source_node rclcpp sensor_msgs cv_bridge OpenCV)

add_executable(visualizer_node src/Visualizer_Node.cpp)
ament_target_dependencies(visualizer_node rclcpp sensor_msgs cv_bridge OpenCV)

# 传统视觉节点
add_executable(traditional_vision_node src/Traditional_Vision_Node.cpp)
# 挂载 ROS2 基础依赖
ament_target_dependencies(traditional_vision_node rclcpp sensor_msgs geometry_msgs std_msgs cv_bridge OpenCV)
# 链接传统视觉算法库与消息支持库
target_link_libraries(traditional_vision_node traditional_detector "${cpp_typesupport_target}")

# 神经网络节点
add_executable(neural_network_node src/Neural_Network_Node.cpp)
ament_target_dependencies(neural_network_node rclcpp sensor_msgs geometry_msgs std_msgs cv_bridge OpenCV)
# 这里非常关键：我们要链接 yolo_detector 库、onnxruntime 引擎、以及自定义消息支持库
target_link_libraries(neural_network_node yolo_detector onnxruntime "${cpp_typesupport_target}")


# ==========================================
# 7. 安装规则
# ==========================================
install(TARGETS
  yolo_detector
  traditional_detector
  test_yolo_local
  test_traditional_local
  image_source_node
  visualizer_node
  traditional_vision_node
  neural_network_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY models DESTINATION share/${PROJECT_NAME})

ament_package()