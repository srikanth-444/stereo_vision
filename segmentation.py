import cv2 as cv
import numpy as np
from cv2 import kmeans
import matplotlib.pyplot as plt

class Segmentation:

    def __init__(self) -> None:
        pass

    def capturing_images(self):
        camera=cv.VideoCapture(0)
        camera.set(cv.CAP_PROP_FRAME_WIDTH,1600)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT,600)
        if not camera.isOpened():
            print("Cannot open camera")
            exit()
        counter=0
        while True:
            ret,frame=camera.read()
            cv.imshow("window",frame)
            if cv.waitKey(1)==ord('c'):
                cv.imwrite('right.png',cv.rotate(frame[:,:800,:],cv.ROTATE_180))
                cv.imwrite('left.png',cv.rotate(frame[:,800:,:],cv.ROTATE_180))
                break
                
    def kmeans_clustering(self,img):
       
        
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        pic=img.reshape(-1,3)
        vectorized=np.float32(pic)

        ret,label,center = kmeans(vectorized,2,None,criteria,2,cv.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image1 = res.reshape(img.shape)
        result_image1=result_image1                           

        

        # plt.imshow(result_image1,'gray')
        # plt.show()
        return result_image1

if __name__=="__main__":
    segmentation=Segmentation()

    segmentation.capturing_images()


    segmentation.kmeans_clustering()