import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from calibration import Calibration
from mpl_toolkits.mplot3d import Axes3D
from open3d import * 
from segmentation import Segmentation

calibration=Calibration()
segmentation=Segmentation()
# calibration.capturing_images()
# calibration.distrotion_remaping("left")
# calibration.distrotion_remaping("right")
 
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
 print("Cannot open camera")
 exit()
while True:
 # Capture frame-by-frame
    if cv.waitKey(1) == ord('q'):
        break
    ret, frame = cap.read()
    
    #grayscale = cv.cvtColor(segmentation.kmeans_clustering(frame), cv.COLOR_BGR2GRAY)
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    half=640

    right_img=cv.rotate(grayscale[:,:half],cv.ROTATE_180)
    left_img=cv.rotate(grayscale[:,half:],cv.ROTATE_180)
    
    left_img,right_img=calibration.re_map(left_img,right_img)
    
    print(left_img.shape,right_img.shape)
    sigma = 1.5
    lmbda = 8000.0
        
    #stereo = cv.StereoBM.create(numDisparities=16*2, blockSize=13)
    stereo2=cv.StereoSGBM.create(minDisparity=0,numDisparities=16*2,blockSize=5,P1 = 10,P2 = 1000,disp12MaxDiff = 0,preFilterCap = 0,uniquenessRatio = 5,speckleWindowSize = 64,speckleRange = 16,)
    stereo3=cv.ximgproc.createRightMatcher(stereo2)
    #disparity = stereo.compute(left_img, right_img)
    disparity2=stereo2.compute(left_img,right_img)
    disparity3=stereo3.compute(right_img,left_img)

    wls_filter = cv.ximgproc.createDisparityWLSFilter(stereo2)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(disparity2, left_img, disparity_map_right=disparity3)
    l_mtx=np.load('left'+"intrinsic.npy")
    r_mtx=np.load('right'+"intrinsic.npy")
    l_dist=np.load('left'+"distortion.npy")
    r_dist=np.load("right"+"distortion.npy")
    r=np.load('r_matrix.npy')
    t=np.load('t_matrix.npy')
    t_n=np.array([1,0,0]).reshape(3,1)
    #print(t,t_n)
    rev_proj_matrix = np.zeros((4,4)) 
    cv.stereoRectify(cameraMatrix1 = l_mtx,cameraMatrix2 = r_mtx,
                  distCoeffs1 = l_dist, distCoeffs2 = r_dist,
                  imageSize = left_img.shape[:2],
                  R = np.identity(3), T = np.array([0.03, 0., 0.]),
                  R1 = None, R2 = None,
                  P1 =  None, P2 =  None, 
                  Q = rev_proj_matrix)
    points = cv.reprojectImageTo3D(filtered_disp, rev_proj_matrix)
    mask_map = filtered_disp > 150
    output_points = points[mask_map]
    color=calibration.center_cropping(cv.rotate(cv.cvtColor(frame, cv.COLOR_BGR2RGB)[:,half:],cv.ROTATE_180))
    print(color.shape)
    color_points=color[mask_map]

    vertices=np.hstack([output_points.reshape(-1,3),color_points.reshape(-1,3)])

    ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
        '''

    with open('pointcloud.ply', 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')
    
    points[points == float('+inf')] = 0
    points[points == float('-inf')] = 0
    cv.imwrite('disparity.png',filtered_disp)
    f,ax= plt.subplots(2,2)
    ax[0,0].imshow(left_img,'gray')
    ax[1,0].imshow(color,'gray')
    ax[0,1].imshow(right_img,'gray')
    ax[1,1].imshow(filtered_disp,'gray')

  
   

    plt.show()
    # cloud = io.read_point_cloud("pointcloud.ply") # Read point cloud
    # visualization.draw_geometries([cloud]) 
    
    
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()



