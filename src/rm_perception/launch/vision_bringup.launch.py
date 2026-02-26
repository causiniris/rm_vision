from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    return LaunchDescription([
        # 1. 启动可视化大屏节点 (尽早启动，建立好 1000 容量的队列等待接收)
        Node(
            package='rm_perception',
            executable='visualizer_node',
            name='visualizer_node',
            output='screen'
        ),
        # 2. 启动预测节点 (接收传统和神经网络，加上绿框后转发)
        Node(
            package='rm_perception',
            executable='predictor_node',
            name='predictor_node',
            output='screen'
        ),
        # 3. 启动传统视觉
        Node(
            package='rm_perception',
            executable='traditional_vision_node',
            name='traditional_vision_node',
            output='screen'
        ),
        # 4. 启动神经网络
        Node(
            package='rm_perception',
            executable='neural_network_node',
            name='neural_network_node',
            output='screen'
        ),
        # 5. 【终极防丢帧杀手锏】：延迟 3 秒启动视频源！
        # 让前面四个兄弟把底层的 ROS 通信管道和 1000 容量的缓冲池彻底建好，再开始发车！
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='rm_perception',
                    executable='image_source_node',
                    name='image_source_node',
                    output='screen'
                )
            ]
        )
    ])