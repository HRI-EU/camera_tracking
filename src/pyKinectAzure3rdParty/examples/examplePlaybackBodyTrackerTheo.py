import cv2

import sys

sys.path.insert(1, "../pyKinectAzure/")

from pyKinectAzure import pyKinectAzure, _k4a

import os

path2SIT = os.getenv("SIT_EXT")

modulePath = os.path.join(path2SIT, "AzureKinect", "lib", "libk4a.so")


if __name__ == "__main__":

    video_filename = "output.mkv"

    # # Initialize the library, if the library is not found, add the library path as argument
    # pyKinectAzure.initialize_libraries(track_body=True)

    # Initialize the library with the path containing the module
    pyK4A = pyKinectAzure(modulePath)

    # Start playback
    playback = pyK4A.start_playback(video_filename)

    playback_config = playback.get_record_configuration()
    # print(playback_config)

    playback_calibration = playback.get_calibration()

    # Start body tracker
    bodyTracker = pyK4A.start_body_tracker(calibration=playback_calibration)

    cv2.namedWindow("Depth image with skeleton", cv2.WINDOW_NORMAL)
    while True:

        # Get camera capture
        ret, capture = playback.update()

        if not ret:
            break

        # Get body tracker frame
        body_frame = bodyTracker.update(capture=capture)

        # Get color image
        ret_color, color_image = capture.get_transformed_color_image()

        # Get the colored depth
        ret_depth, depth_color_image = capture.get_colored_depth_image()

        # Get the colored body segmentation
        ret_seg, body_image_color = body_frame.get_segmentation_image()

        if not ret_color or not ret_depth or not ret_seg:
            continue

        # Combine both images
        combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
        combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, combined_image, 0.3, 0)

        # Draw the skeletons
        combined_image = body_frame.draw_bodies(combined_image)

        # Overlay body segmentation on depth image
        cv2.imshow("Depth image with skeleton", combined_image)

        # Press q key to stop
        if cv2.waitKey(1) == ord("q"):
            break
