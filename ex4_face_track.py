#! /usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import matplotlib.pyplot as plt
import sys
from djitellopy import Tello

tello = Tello()

class Tello(Node):
    def __init__(self):
        super().__init__('tello')
        
        self.subscription = self.create_subscription(
            Image,
            '/camera',
            self.camera_callback,
            10
        )

        self.publisher_ = self.create_publisher(Twist, '/drone1/cmd_vel', 10)
    
        self.face_pixels = []

    def camera_callback(self, msg):

        width = 12.0
        dist = 40.0
        self.face_width = 0
        self.ref_w = 0

        tello.takeoff()

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.img = self.imgmsg_to_cv2(msg)
        self.ref_img = self.imgmsg_to_cv2(msg)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.ref_img = cv2.imread("ref-img.png")
        ref_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        ref_img = face_cascade.detectMultiScale(ref_gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            self.face_width = w
        

        for (x, y, w, h) in ref_img:
            cv2.rectangle(self.ref_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            self.ref_w = w

        ref_flength = (dist * self.ref_w) / width
        camera_dist = (ref_flength * width) / self.face_width
        cv2.putText(self.img, f"Distance: {round(camera_dist,2)} CM", (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(self.img, f"Face: {round(self.face_width,2)} PIXELS", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        self.face_pixels.append(self.face_width)
            
        cv2.imshow('img', self.img)
        cv2.waitKey(1)
        
        if self.face_pixels == "inf":
            tello.rotate_clockwise(90)

        cmd_publish = Twist()
        cmd_publish.angular.z = 1.0
        self.publisher_.publish(cmd_publish)
        


    def imgmsg_to_cv2(self, img_msg):

        n_channels = len(img_msg.data) // (img_msg.height * img_msg.width)
        dtype = np.uint8

        img_buf = np.asarray(img_msg.data, dtype=dtype) if isinstance(img_msg.data, list) else img_msg.data

        if n_channels == 1:
            cv2_img = np.ndarray(shape=(img_msg.height, img_msg.width), dtype=dtype, buffer=img_buf)
        else:
            cv2_img = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels), dtype=dtype, buffer=img_buf)

        # If the byte order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            cv2_img = cv2_img.byteswap().newbyteorder()

        return cv2_img
    
    

def main(args=None):
    rclpy.init(args=args)
    tello = Tello()
    rclpy.spin(tello) 
    tello.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()