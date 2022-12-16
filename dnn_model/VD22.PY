# Import opencv
import cv2
# Import operating sys
import os
# Import matplotlib
from matplotlib import pyplot as plt
# Establish capture
cap = cv2.VideoCapture('G:\course_320\视频素材参考\CF.mp4')

# Properties
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
# Video Writer
video_writer = cv2.VideoWriter(os.path.join('G:\course_320\视频素材参考' ,'output' ,'output.avi'), cv2.VideoWriter_fourcc('P' ,'I' ,'M' ,'1'), fps, (width, height), isColor=False)

# Loop through each frame
for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):

    # Read frame
    ret, frame = cap.read()

    # Gray transform
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Show image
    cv2.imshow('Video Player', gray)

    # Write out frame
    video_writer.write(gray)

    # Breaking out of the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Close down everything
cap.release()
cv2.destroyAllWindows()
# Release video writer
video_writer.release()