import cv2 as cv
import numpy as np

# Camera setup
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 30)  # request 30 FPS

# Check actual FPS
fps = cap.get(cv.CAP_PROP_FPS)
print(f"Camera FPS: {fps}")

# Video writer setup
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'MJPG'
out = cv.VideoWriter('output_video.mp4', fourcc, fps, (1280, 480))

# Optional matrix
rev_proj_matrix = np.zeros((4,4))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Write frame to video
    out.write(frame)

    # Optional: show live feed
    cv.imshow('Camera', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv.destroyAllWindows()
