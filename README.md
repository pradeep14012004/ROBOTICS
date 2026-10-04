# Robotics

A collection of robotics and ROS 2 experiments, controllers and simulation work developed while learning autonomous robotics.

## Focus

- ROS 2
- Robot control
- Gazebo simulation
- Sensor-based behaviour
- Navigation and SLAM
- Python-based robotics development

## Environment

- Ubuntu 22.04
- ROS 2 Humble
- Python
- Gazebo
- colcon

## Typical ROS 2 Workflow

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build
source install/setup.bash
ros2 launch <package> <launch_file>
```

## Project Organization

As experiments are added, each robotics exercise should contain its own source, launch/configuration files and README.

```text
ROBOTICS/
├── packages/
├── simulations/
├── controllers/
├── launch/
├── config/
├── docs/
└── README.md
```

## Goal

Build practical experience in robot control, simulation, perception and autonomous navigation rather than maintaining isolated tutorial code.


## Browser Demo

A browser-accessible differential-drive kinematics simulator is available under `web/`. It runs without ROS 2 or Gazebo and supports keyboard control, straight motion and circular motion.

```bash
docker compose -f docker-compose.web.yml up --build
```

Open `http://localhost:8080`.