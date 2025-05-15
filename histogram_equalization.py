import cv2
import argparse


def histogram_equalization(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    # Convert back to BGR
    equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return equalized_bgr


def process_video(input_path, output_path):
    # Open the input video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec

    # Create the output video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply histogram equalization
        equalized_frame = histogram_equalization(frame)

        # Write the frame to the output file
        out.write(equalized_frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Histogram equalized video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply histogram equalization to a video.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_video", help="Path to save the histogram equalized video")
    args = parser.parse_args()

    process_video(args.input_video, args.output_video)
