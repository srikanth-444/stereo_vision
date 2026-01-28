import cv2
from slam2 import ORB_SLAM
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
video_path = 'output_video.mp4'
l_k='camera_mat/leftintrinsic.npy'
r_k='camera_mat/rightintrinsic.npy'
l_dist='camera_mat/leftdistortion.npy'
r_dist='camera_mat/rightdistortion.npy'
R='camera_mat/r_matrix.npy'
t='camera_mat/t_matrix.npy'
frame_id=5

slam = ORB_SLAM(l_k, r_k, l_dist, r_dist, R, t)
slam2 = ORB_SLAM(l_k, r_k, l_dist, r_dist, R, t)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

ret, frame = cap.read()
cap.release()

right_img = frame[:, :frame.shape[1] // 2]
left_img = frame[:, frame.shape[1] // 2:]

gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
gray_left = cv2.rotate(gray_left, cv2.ROTATE_180)
gray_right = cv2.rotate(gray_right, cv2.ROTATE_180)

gray_left = slam.clahe.apply(gray_left)
gray_right = slam.clahe.apply(gray_right)
slam.create_new_landmarks(gray_left, gray_right,nonlinear_triangulation=False)
lx= [lm.position[0] for lm in slam.land_marks]
ly = [lm.position[1] for lm in slam.land_marks]
lz = [lm.position[2] for lm in slam.land_marks]
slam2.create_new_landmarks(gray_left, gray_right,nonlinear_triangulation=True)
gx= [lm.position[0] for lm in slam2.land_marks]
gz = [lm.position[2] for lm in slam2.land_marks]
vis = np.hstack((gray_left, gray_right))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot first set of points in green
ax.scatter(lx, ly, lz, c='g', s=5, alpha=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for lm in slam.land_marks:
    if not lm.active:
        continue
    x, y = int(lm.image_points[0]), int(lm.image_points[1])
    x_r, y_r = int(lm.r_image_points[0])+gray_left.shape[1], int(lm.r_image_points[1])
    cv2.circle(vis, (x,y), 3, (255,0,0), -1)
    cv2.circle(vis, (x_r,y_r), 3, (255,0,0), -1)
    cv2.putText(vis, f"{lm.id}", (x+3,y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)
    cv2.putText(vis, f"{lm.id}", (x_r+3,y_r-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1)
cv2.circle(vis, (int(slam.cam.l_K[0,2]), int(slam  .cam.l_K[1,2])), 5, (0,0,255), -1)  # just a marker for visualization
cv2.putText(vis, f"Frame: {slam.count_frame}, Tracked: {len([lm for lm in slam.land_marks if lm.active])}, Landmarks: {len(slam.land_marks)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
cv2.imshow("Tracked", vis)

cv2.waitKey(0)

plt.show()
cv2.destroyAllWindows()
