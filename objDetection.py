import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import re

font = cv2.FONT_HERSHEY_SIMPLEX

col_frames = os.listdir('frames/')
col_frames.sort(key = lambda f : int(re.sub('\D','',f)))

col_images = []
for i in col_frames :
	img = cv2.imread('frames/'+i)
	col_images.append(img)

pathOut = 'vehicle_detection.mp4'	
frame_array = []

for i in range(len(col_images)-1) :
	grayA = cv2.cvtColor(col_images[i], cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(col_images[i+1], cv2.COLOR_BGR2GRAY)

	diff_image = cv2.absdiff(grayB, grayA)
	ret, thresh = cv2.threshold(diff_image, 35, 255, cv2.THRESH_BINARY)

	kernel = np.ones((3,3),np.uint8)
	dilated = cv2.dilate(thresh,kernel,iterations = 2)

	contours, hierarchy = cv2.findContours(dilated.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

	valid_contours = []
	for ctr in contours :
		x,y,w,h = cv2.boundingRect(ctr)
		if (y >= 80) and (y <= 130) and (cv2.contourArea(ctr) > 100) :
			valid_contours.append(ctr)

	dummy = col_images[i].copy()
	cv2.drawContours(dummy, valid_contours, -1, (0,0,0), 1)
	cv2.putText(dummy, "vehicles detected: " + str(len(valid_contours)), (55, 15), font, 0.4, (0, 0, 0), 2)		
	
	frame_array.append(dummy)

h, w, l = frame_array[0].shape	
size = (w, h)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'mp4v'), 14, size)
for i in range(len(frame_array)) :
	out.write(frame_array[i])

out.release()




