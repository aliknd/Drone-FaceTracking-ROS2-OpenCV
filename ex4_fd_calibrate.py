#! /usr/bin/python
import base64
import cv2 as cv
import numpy as np
import imutils
import matplotlib.pyplot as plt

import time
import rclpy
from rclpy.node import Node
import sys
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rosgraph_msgs.msg import Clock

class CamSubscriber(Node):
    def __init__(self):
        super().__init__('image_listener')
        self.cnt = 0
        self.cnt2 = 0

        self.image_sub = self.create_subscription(Image, '/camera', self.image_sub_callback, 10)

        #super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'face_found', 10)


    def image_sub_callback(self, msg):

        # Convert ROS Image message to OpenCV2
        cv2_img = self.imgmsg_to_cv2(msg)

        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml') 
        gray = cv.cvtColor(cv2_img, cv.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
        for (x, y, self.fw, self.fh) in faces: 
            face = cv.rectangle(cv2_img, (x, y), (x+self.fw, y+self.fh), (255, 0, 0), 2)
            cv.putText(cv2_img, 'face', (x+self.fw, y+self.fh), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            self.cnt += 1 
        cv.imwrite('face_image_{}.jpeg'.format(self.cnt), cv2_img)

        if self.cnt > 0:
            msg = String()
            msg.data = f'A face has been found with the size of width: {self.fw} px and height: {self.fh} px: %d' % self.cnt
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)

    def imgmsg_to_cv2(self, img_msg):
        n_channels = len(img_msg.data) // (img_msg.height * img_msg.width)
        dtype = np.uint8

        img_buf = np.asarray(img_msg.data, dtype=dtype) if isinstance(img_msg.data, list) else img_msg.data

        if n_channels == 1:
            cv2_img = np.ndarray(shape=(img_msg.height, img_msg.width),
                            dtype=dtype, buffer=img_buf)
        else:
            cv2_img = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                            dtype=dtype, buffer=img_buf)

        # If the byte order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            cv2_img = cv2_img.byteswap().newbyteorder()

        return cv2_img


def main():
    rclpy.init()
    cam_subscriber = CamSubscriber()

    # Spin until ctrl + c
    rclpy.spin(cam_subscriber)

    cam_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()