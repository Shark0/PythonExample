import cv2
import os


def video_to_frames(video_path, output_dir, interval=1, image_format="jpg"):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")

    # Counter for saved frames
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame if it matches the interval
        if frame_count % interval == 0:
            # Generate output file path
            output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.{image_format}")
            # Save frame as image
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    # Release video capture
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")


def main():
    # Input parameters
    video_path = "example.mp4"  # Replace with your video path
    output_dir = "output_frames"  # Directory to save frames
    interval = 1  # Save every frame (1 = all frames, 30 = every 30th frame)
    image_format = "jpg"  # Save as JPG (or 'png' for higher quality)

    # Extract frames
    video_to_frames(video_path, output_dir, interval, image_format)

if __name__ == "__main__":
    main()