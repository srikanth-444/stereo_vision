import cv2
import numpy as np
import tensorflow as tf


model = tf.saved_model.load("model_dir")
infer = model.signatures["serving_default"]

# =========================
# Load images
# =========================
imgL = cv2.imread("/home/srikanth/machine_hall/machine_hall/MH_01_easy/MH_01_easy/mav0/cam0/data/1403636579763555584.png")
imgR = cv2.imread("/home/srikanth/machine_hall/machine_hall/MH_01_easy/MH_01_easy/mav0/cam1/data/1403636579763555584.png")

h, w = imgL.shape[:2]

# =========================
# Intrinsics
# =========================
K1 = np.array([[458.654, 0, 367.215],
               [0, 457.296, 248.375],
               [0, 0, 1]], dtype=np.float64)

K2 = np.array([[457.587, 0, 379.999],
               [0, 456.134, 255.238],
               [0, 0, 1]], dtype=np.float64)

# =========================
# Distortion (must be 5 params)
# =========================
D1 = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05, 0.0])
D2 = np.array([-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05, 0.0])

# =========================
# Extrinsics (your inputs)
# =========================
e1 = np.array([
    0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
    0.999598781151,  0.0130119051815, 0.0251588363115,  0.0453689425024,
   -0.0253898008918, 0.0179005838253, 0.999517347078,  0.00786212447038,
    0.0, 0.0, 0.0, 1.0
]).reshape(4,4)

e2 = np.array([
    0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
    0.999557249008,  0.0149672133247, 0.025715529948,   -0.064676986768,
   -0.0257744366974, 0.00375618835797, 0.999660727178,  0.00981073058949,
    0.0, 0.0, 0.0, 1.0
]).reshape(4,4)

# =========================
# Compute relative pose (IMPORTANT)
# =========================
R1 = e1[:3, :3]
t1 = e1[:3, 3]

R2 = e2[:3, :3]
t2 = e2[:3, 3]

# Relative transform: cam1 -> cam2
R = R2 @ R1.T
T = t2 - R @ t1

print("Baseline:", np.linalg.norm(T))

# =========================
# Stereo Rectification
# =========================
R1_rect, R2_rect, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1,
    K2, D2,
    (w, h),
    R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

# =========================
# Rectification maps
# =========================
map1L, map2L = cv2.initUndistortRectifyMap(
    K1, D1, R1_rect, P1, (w, h), cv2.CV_32FC1
)

map1R, map2R = cv2.initUndistortRectifyMap(
    K2, D2, R2_rect, P2, (w, h), cv2.CV_32FC1
)

# =========================
# Apply rectification
# =========================
rectL = cv2.remap(imgL, map1L, map2L, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map1R, map2R, cv2.INTER_LINEAR)

left = cv2.resize(rectL, (w, h))
right = cv2.resize(rectR, (w, h))


left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

left = left.astype(np.float32) / 255.0
right = right.astype(np.float32) / 255.0

left = np.expand_dims(left, axis=0)
right = np.expand_dims(right, axis=0)
outputs = infer(left=left, right=right)
disp = list(outputs.values())[0].numpy()[0]
disp_vis = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)

cv2.imshow("Disparity", disp_vis)
cv2.waitKey(0)