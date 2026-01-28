import numpy as np
import cv2 as cv

import os 


class Calibration():
    def __init__(self) -> None:
        self.x=0
        self.y=0
        self.h=0
        self.w=0
     
    def capturing_images(self,num):
        camera=cv.VideoCapture(2)
        camera.set(cv.CAP_PROP_FRAME_WIDTH,1280)
        camera.set(cv.CAP_PROP_FRAME_HEIGHT,480)
        if not camera.isOpened():
            print("Cannot open camera")
            exit()
        counter=0
        while True:
            ret,frame=camera.read()
            cv.imshow("window",frame)
            if cv.waitKey(1)==ord('c'):
                cv.imwrite('right/image'+str(counter)+'.png',cv.rotate(frame[:,:640,:],cv.ROTATE_180))
                cv.imwrite('left/image'+str(counter)+'.png',cv.rotate(frame[:,640:,:],cv.ROTATE_180))
                counter=counter+1
                if counter==num:
                    break
    def single_camera_calibration(self,dir):
        
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((9*5,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:5].T.reshape(-1,2)
        objp = objp*24  # Assuming square size of 25mm
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        
        images = dir
        
        for fname in os.listdir(images):
            img = cv.imread(dir+'/'+fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (9,5), None)


            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
            
            # Draw and display the corners
                cv.drawChessboardCorners(img, (9,5), corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(500)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(mtx)
        np.save(dir+"intrinsic.npy",mtx)
        np.save(dir+"distortion.npy",dist)
        img = cv.imread(dir+'/image0.png')
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        # undistort
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv.imwrite(dir+'calibresult.png', dst)
        
        cv.destroyAllWindows()

    def stereo_camera_calibration(self):
        l_mtx=np.load('left'+"intrinsic.npy")
        r_mtx=np.load('right'+"intrinsic.npy")
        l_dist=np.load('left'+"distortion.npy")
        r_dist=np.load("right"+"distortion.npy")


        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

        objp = np.zeros((9*5,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:5].T.reshape(-1,2)
        objp = objp*24.0  # Assuming square size of 30mm

        objpoints = []
        left_imgpoints = []
        right_imgpoints= []
        
        dir='left'
        
        for fname in os.listdir(dir):
            img = cv.imread(dir+'/'+fname)
            img_size = (img.shape[1], img.shape[0])
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            

            ret, corners = cv.findChessboardCorners(gray, (9,5), None)
            
     
            if ret == True:
                objpoints.append(objp)
                
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                left_imgpoints.append(corners)
            
  
                cv.drawChessboardCorners(img, (9,5), corners2, ret)
                cv.imshow('img', img)
                cv.waitKey(500)

        dir='right'
        for fname in os.listdir(dir):
                img = cv.imread(dir+'/'+fname)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            

                ret, corners = cv.findChessboardCorners(gray, (9,5), None)
            
     
                if ret == True:
                   
                
                    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                    right_imgpoints.append(corners)


                    cv.drawChessboardCorners(img, (9,5), corners2, ret)
                    cv.imshow('img', img)
                    cv.waitKey(500)

        _, _, _, _, _, rotationMatrix, translationVector, _, _ = cv.stereoCalibrate(
                                                                                objpoints, left_imgpoints, right_imgpoints,
                                                                                        l_mtx, l_dist,r_mtx, r_dist,
        img_size, None, None, None, None,
       criteria = criteria,flags=cv.CALIB_FIX_INTRINSIC)

        np.save('r_matrix.npy',rotationMatrix)
        np.save('t_matrix.npy',translationVector)
        print(rotationMatrix, translationVector)

    def re_map(self,img,dir):
        h, w = img.shape
        self.mtx=np.load(dir+"intrinsic.npy")
        self.dist=np.load(dir+"distortion.npy")
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
       
        mapx, mapy = cv.initUndistortRectifyMap(self.mtx, self.dist, None, newcameramtx, (w,h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
        
        
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        print(dst.shape)
        return dst
    def re_map(self,l_img,r_img):
        h, w = l_img.shape
        l_mtx=np.load("leftintrinsic.npy")
        l_dist=np.load("leftdistortion.npy")
        r_mtx=np.load("leftintrinsic.npy")
        r_dist=np.load("leftdistortion.npy")
        leftcameramtx, l_roi = cv.getOptimalNewCameraMatrix(l_mtx,l_dist, (w,h), 1, (w,h))
        rightcameramtx, r_roi = cv.getOptimalNewCameraMatrix(r_mtx,r_dist, (w,h), 1, (w,h))
        l_mapx, l_mapy = cv.initUndistortRectifyMap(l_mtx, l_dist, None, leftcameramtx, (w,h), 5)
        r_mapx, r_mapy = cv.initUndistortRectifyMap(r_mtx, r_dist, None, rightcameramtx, (w,h), 5)
        l_dst = cv.remap(l_img, l_mapx, l_mapy, cv.INTER_LINEAR)
        r_dst = cv.remap(r_img, r_mapx, r_mapy, cv.INTER_LINEAR)
        
        
        l_x, l_y, l_w, l_h = l_roi
        r_x, r_y, r_w, r_h = r_roi

        self.x=max(l_x,r_x)
        self.y=max(l_y,r_y)
        self.h=min(l_h,r_h)
        self.w=min(l_w,r_w)


        l_dst = l_dst[self.y:self.y+self.h, self.x:self.x+self.w]
        r_dst = r_dst[self.y:self.y+self.h, self.x:self.x+self.w]
        
        return l_dst,r_dst
 
    
    def center_cropping(self,img):
        img=img[self.y:self.y+self.h, self.x:self.x+self.w]
        return img
if __name__=="__main__":
    calibration=Calibration()
    # calibration.capturing_images(10)
    calibration.single_camera_calibration('left')
    calibration.single_camera_calibration("right")
    calibration.stereo_camera_calibration()