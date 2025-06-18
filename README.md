# Unitree OVMM Setup and Code Guide

## Overview
This repository contains code for working with Unitree Go2 robots. It includes setup instructions, running procedures, and solutions to common issues encountered during development.

## Prerequisites
- Ubuntu tested on WSL Ubuntu 22.04
- Unitree SDK
- Unitree SDK for python

## Installation OVMM
```sh
git clone https://github.com/ai4ce/OVMM.git
cd OVMM
conda create --name OVMM python=3.10
conda activate OVMM
pip install -r requirements.txt
```

## Installation Unitree SDK
### 1. Clone the Repository
```sh
git clone https://github.com/unitreerobotics/unitree_sdk2.git
```

### 2. Install Unitree SDK
```sh
cd ~/unitree_sdk2/
mkdir build
cd build
cmake ..
sudo make install
```

### 3. Compile Examples
```sh
make
```

## Install Unitree SDK Python
### Dependencies
- python >= 3.8
- cyclonedds == 0.10.2
- numpy
- opencv-python

### Install Unitree SDK Python
```sh
cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```

#### Error when running `pip3 install -e .`
```sh
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

#### This error indicates that the path to `cyclonedds` cannot be found. First, compile and install `cyclonedds`:
```sh
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_DDSPERF=OFF
cmake --build . --target install
```

#### Set `CYCLONEDDS_HOME` and install `unitree_sdk2_python` Navigate to the  Unitree_sdk2_python directory, set `CYCLONEDDS_HOME` to the newly compiled `cyclonedds` path, and install `unitree_sdk2_python`:
```sh
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME="/{path to your cyclonedds}/cyclonedds/install"
pip3 install -e .
```

## Configure Network Environment
When running example programs, control commands will be sent from the user's computer to the Go2 robot's onboard computer via a local network. Therefore, before proceeding, some necessary configuration steps are required to establish a local network between the two computers.
### Configuration Steps:
1. Connect the network cable
    - Connect one end of the Ethernet cable to the Go2 robot and the other end to the user's computer.
    - Enable the USB Ethernet interface on the computer and configure its network settings.
    - The Go2 robot's onboard computer has a fixed IP address of 192.168.123.161.
    - The computer's USB Ethernet address should be set to the same subnet, such as 192.168.123.222 (you can replace "222" with another number).
    ![image](https://github.com/user-attachments/assets/3216e80d-db1c-4beb-8a91-871b13df3da4)
    ![image](https://github.com/user-attachments/assets/07c78157-8621-4e8d-8ad6-f573170dc11f)


2. Verify the connection
    - To check if the user's computer is properly connected to the Go2 robot’s onboard computer, run the command `ping 192.168.123.161` in the terminal.
    - If the output is similar to the following, the connection is successful:
    ![image](https://github.com/user-attachments/assets/e7ae3356-eda9-4d89-aae3-8d0c20520d3d)



3. Check the network interface name for the 123 subnet
    - Use the ifconfig command to find the name of the network interface corresponding to the 192.168.123.xxx subnet, as shown in the example below:
    ![image](https://github.com/user-attachments/assets/64eeb5d0-26dd-41a0-8121-d1882511a8f3)


    - In the output, the network interface name corresponding to the IP `192.168.123.222` is `enxf8e43b808e06`.
    - The user needs to remember this name, as it will be required as a parameter when running example programs.

## Running example programs
Open a terminal and execute the following commands in sequence to run the example program.
Note: In the second command, replace the string `enxf8e43b808e06` with the actual network interface name corresponding to the 123 subnet on your computer.

```sh
cd ~/unitree_sdk2_python/example/go2/high_level
python go2_sport_client.py
```

Note:
When prompted with "Enter id or name:", entering 0 will switch the robot to damp mode, which helps save energy.
You can enter 1 afterward to make the robot stand up again.

## Running MAST3R with Go2 Robot
Run the MAST3R post estimation api on workstation before running `main.py` on Go2 Robot.
### 1. Download and install MAST3R on workstation
Follow the installation step on [MAST3R](https://github.com/naver/mast3r) repo
 
### 2. Run post estimation API
After the installation, run `post_estimation_api.py` on workstation.

### 3. Run Go2 Robot
After turn on the post estimation API, run the `main.py` on Go2 Robot

## Future Improvements
- Add motion planning algorithms
- Add mobile manipulation

## Contributors
- **Chi-Feng, Liu** (cl6933@nyu.edu)
- **Mentor: [Juexiao Zhang](https://juexzz.github.io/)** 
