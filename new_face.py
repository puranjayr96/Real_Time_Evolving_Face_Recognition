import cv2
import os
vidcap = cv2.VideoCapture(0)
name = input("What's your name?")
datadir = './input_dir'
new_file = os.path.join(datadir, name)
if not os.path.exists(new_file):
    os.makedirs(new_file)
success,image = vidcap.read()
count = 0
success = True
while success:
    success,image = vidcap.read()
    if(count%3==0):
        img_num = count/3
        cv2.imwrite("./input_dir/%s/%s%d.jpg" % (name, name,img_num), image)     # save frame as JPEG file
    cv2.imshow('Video', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count += 1

vidcap.release()

cv2.destroyAllWindows()

