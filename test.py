import cv2
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\aryan\Downloads\AGV TASK\LK.mp4")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

#
output_path = "tracked_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'avc1')  
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

f_params = dict(maxCorners=90, qualityLevel=0.1, minDistance=9, blockSize=7)

lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 8, 0.03))

ret, old_frame = cap.read()
if not ret:
    print("Error: Cannot read video")
    cap.release()
    cv2.destroyAllWindows()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **f_params)

mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 0, 255), 1)  
            cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)  

        output = cv2.add(frame, mask)

        print(f"Tracked Points: {len(good_new)}")

        if len(good_new) < 10:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **f_params)
            mask = np.zeros_like(frame)  
            print("Re-detecting feature points...")

    cv2.imshow('Tracking', output)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {r"C:\Users\aryan\Downloads\AGV"}")
