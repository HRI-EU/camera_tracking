#!/usr/bin/env python
import time
import cv2
from camera_tracking.pykinect_azure_fix import pykinect_azure as pykinect


if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries(track_body=True)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30

    # Start device
    device = pykinect.start_device(config=device_config)

    # Start body tracker
    bodyTracker = pykinect.start_body_tracker()

    # Initialize statistics.
    step_count = 0
    sum_overall_time = 0.0
    sum_capture_time = 0.0
    report_interval = 20

    while True:
        start_time = time.time()

        # Get capture
        capture = device.update()
        sum_capture_time += time.time() - start_time

        # Get body tracker frame
        body_frame = bodyTracker.update()

        # Get the color depth image from the capture
        ret_depth, depth_color_image = capture.get_colored_depth_image()

        # Get the colored body segmentation
        ret_color, body_image_color = body_frame.get_segmentation_image()

        if not ret_depth or not ret_color:
            continue

        # Combine both images
        combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)

        # Draw the skeletons
        combined_image = body_frame.draw_bodies(combined_image)

        # Overlay body segmentation on depth image
        cv2.namedWindow("Depth image with skeleton", cv2.WINDOW_NORMAL)
        cv2.imshow("Depth image with skeleton", combined_image)

        # Press q key to stop
        if cv2.waitKey(1) == ord("q"):
            break

        sum_overall_time += time.time() - start_time
        step_count += 1
        if step_count % report_interval == 0:
            status = (
                f"Step {step_count} / "
                f"Mean capture time: {sum_capture_time / report_interval:.4f}s / "
                f"Mean processing time: {(sum_overall_time - sum_capture_time) / report_interval:.4f}s"
            )

            print(status)
            sum_overall_time = 0.0
            sum_capture_time = 0.0
