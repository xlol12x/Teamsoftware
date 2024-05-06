Linux Requirements:
opencv-python
mediapipe
tensorflow[and-cuda]
pynput
ROS Melodic (Ubuntu 18.04) {for robot simulation}
Champ_msgs ROS Package {for robot simulation}

Image on docker hub (linux only): https://hub.docker.com/repository/docker/bahtes/gestures-to-keypress

To run docker image:

xhost +local:docker
sudo docker run -it --device /dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY bahtes/gestures-to-keypress

(camera needed to change between webcams change "/dev/video0:/dev/video0" to the corrosponding device)

720p camera for best experience
reccomend camo if no webcam present
https://reincubate.com/camo/

for controls:
right hand controlls movement, in red box is the deadzone
right hand only track on right of green line
left hand only tracked on the left of the green line
use config settings to select what gestures input
select either keyboard or controller emulation


base of middle finger used for hand position 

file- gesture.names and folder gestures in same dictionary
