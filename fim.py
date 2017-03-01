from glob import glob
import PIL 
import scipy.misc
import cv2
eye_cascade = cv2.CascadeClassifier('C:\\Users\\pratik\\Desktop\\Face detection\\haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('C:\\Users\\pratik\\Desktop\\Face detection\\haarcascade_frontalface_default.xml')
img_mask = 'C:/Users/pratik/Desktop/Face detection/Img/*.jpg'
img_names = glob(img_mask)
count=0
for fn in img_names:
    print('processing %s...' % fn,)
    img = cv2.imread(fn, 0)
    
    faces= face_cascade.detectMultiScale(img,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        face=img[x,y]
        roi_gray = img[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        cv2.imwrite("C:/Users/pratik/Desktop/Face detection/Image/face%d.jpg" % count,roi_gray)
        count += 1

cv2.destroyAllWindows()
