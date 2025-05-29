import cv2
import os


def extract_frames(video_path, desired_fps=10):
    # output_directory=f'Frames_{video_path}'
    output_directory = "frames"
    os.makedirs(output_directory, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = int(fps / desired_fps)
        frame_count = 0
        image_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                image_count += 1
                frame_filename = os.path.join(
                    output_directory, f"{image_count:04d}.jpg"
                )
                cv2.imwrite(frame_filename, frame)

            frame_count += 1
        print("Number of frames extracted is ", image_count)
    else:
        print("err")
    cap.release()
    cv2.destroyAllWindows()
    return image_count


def scale_down_frames(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_files = [
        f
        for f in os.listdir(input_folder)
        if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    scale_factor = 0.25

    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        img = cv2.imread(input_path)

        height, width, _ = img.shape

        # Calculate the new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image
        resized_img = cv2.resize(
            img, (new_width, new_height), interpolation=cv2.INTER_AREA
        )
        # Save the resized image
        cv2.imwrite(output_path, resized_img)
