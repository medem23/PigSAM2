import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from sam2.build_sam import build_sam2_video_predictor
import os, glob
import tqdm
from matplotlib.figure import Figure
import pandas as pd
from io import BytesIO
import zipfile
import tempfile
from pathlib import Path
import subprocess


st.set_page_config(page_title="Pig Analysis", layout="wide")
# ---------------------
# Authentication Setup
# ---------------------
# In a real scenario, store these securely, use environment variables, or integrate with a user database.
VALID_USERNAME = "user"
VALID_PASSWORD = "pass"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "output_dir" not in st.session_state:
    st.session_state.output_dir = os.path.join(os.getcwd(), "tmp_folder")

if "step" not in st.session_state:
    st.session_state.step = 1

if "load_frames" not in st.session_state:
    st.session_state.load_frames = True

if "clean_folder" not in st.session_state:
    st.session_state.clean_folder = True

if "prompts_dict" not in st.session_state:
    st.session_state.prompts_dict = {}

if "bounding_boxes" not in st.session_state:
    st.session_state.bounding_boxes = []

if "temp_video_path" not in st.session_state:
    st.session_state.temp_video_path = None

if "frame_rgb" not in st.session_state:
    st.session_state.frame_rgb = None

if "center_points" not in st.session_state:
    st.session_state.center_points = None

if "video_output_path" not in st.session_state:
    st.session_state.video_output_path = None
    
if "results_zip" not in st.session_state:
    st.session_state.results_zip = None

# New session state for incremental reprocessing:
if "reprocess_from_time" not in st.session_state:
    st.session_state.reprocess_from_time = 0

# Global Start Over and Logout Buttons
st.sidebar.header("Navigation")

if st.sidebar.button("Start Over"):
    st.session_state.step = 1
    st.session_state.bounding_boxes = []
    st.session_state.prompts_dict = {}
    st.session_state.clean_folder = True
    st.session_state.load_frames = True
    st.session_state.frame_rgb = None
    st.session_state.temp_video_path = None
    st.session_state.center_points = []
    st.session_state.video_output_path = None
    st.rerun()

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.step = 1
    st.session_state.bounding_boxes = []
    st.session_state.prompts_dict = {}
    st.session_state.clean_folder = True
    st.session_state.load_frames = True
    st.session_state.frame_rgb = None
    st.session_state.temp_video_path = None
    st.session_state.center_points = []
    st.session_state.video_output_path = None
    st.rerun()

# Some preliminary settings
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load model
sam2_checkpoint = os.path.join(os.getcwd(), "checkpoints", "sam2_hiera_base_plus.pt")
model_cfg = "sam2_hiera_b+.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


def save_video_frames_as_images(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Capture the video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    interval = max(1, frame_count // 20)

    # Initialize frame index
    frame_index = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Convert from BGR to YCrCb
        frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # 2. Split the YCrCb image into its channels
        y_channel, cr_channel, cb_channel = cv2.split(frame_ycrcb)

        # 3. Create a CLAHE object (tweak clipLimit and tileGridSize as needed)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # 4. Apply CLAHE on the Y channel
        y_channel_clahe = clahe.apply(y_channel)

        # 5. Merge back the enhanced Y channel with the original Cr and Cb channels
        frame_clahe_ycrcb = cv2.merge((y_channel_clahe, cr_channel, cb_channel))

        # 6. Convert from YCrCb to RGB
        frame_clahe_rgb = cv2.cvtColor(frame_clahe_ycrcb, cv2.COLOR_YCrCb2RGB)

        # Format the frame filename with zero-padded index
        frame_filename = os.path.join(output_dir, f"frame_{frame_index:06d}.jpg")

        # Save the frame as an image
        cv2.imwrite(frame_filename, frame_rgb)
        frame_index += 1

        # Print progress at specified intervals
        if frame_index % interval == 0 or frame_index == frame_count:
            progress_percent = (frame_index / frame_count) * 100
            print(f"Progress: {progress_percent:.2f}% ({frame_index}/{frame_count} frames)")

    # Release the video capture object
    cap.release()
    print(f"Frames saved to {output_dir} as images successfully.")


def keep_largest_connected_component(binary_mask):
    # Get connected components and their stats
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    if num_labels <= 1:
        return binary_mask  # No components found, return original mask

    # Find the label with the largest area (ignore background label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create a mask for the largest component
    return (labels == largest_label).astype(np.uint8)


def delete_images_from_folder(folder_path, extensions=("*.jpg", "*.png", "*.jpeg")):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Iterate over the specified extensions and delete matching files
    deleted_files_count = 0
    for ext in extensions:
        for file_path in glob.glob(os.path.join(folder_path, ext)):
            try:
                os.remove(file_path)
                deleted_files_count += 1
            except OSError as e:
                pass


def chunk_sequence(image_list, chunk_size):
    # Create an empty list to store the subsequences
    subsequences = []

    # Loop through the list, stepping by chunk_size each time
    for i in range(0, len(image_list), chunk_size):
        subsequences.append(image_list[i:i + chunk_size])

    return subsequences


def get_bbox(gt2D):
    y_indices, x_indices = np.where(gt2D > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = gt2D.shape
    x_min = max(0, x_min)
    x_max = min(W, x_max)
    y_min = max(0, y_min)
    y_max = min(H, y_max)
    bboxes = np.array([x_min, y_min, x_max, y_max])

    return bboxes


def calculate_center(mask):
    moments = cv2.moments(mask)
    if moments["m00"] > 0:  # To avoid division by zero
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy
    return None, None


def login_screen():
    st.title("Login")
    username = st.text_input("Username").strip()
    password = st.text_input("Password", type="password").strip()
    if st.button("Login"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password")


def step_2_annotation():
    st.subheader("Specify a time and annotate the video frame")

    time_sec = st.number_input("Time in seconds to load for annotation:", min_value=0, value=0, step=1)

    # Button to load that frame
    if st.button("Load Frame"):
        cap = cv2.VideoCapture(st.session_state.temp_video_path)
        # Jump to the desired time in milliseconds
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        ret, frame = cap.read()
        cap.release()

        if ret:
            st.session_state.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            st.error("Could not read frame at that time. Try another second.")

    if st.session_state.frame_rgb is not None:
        st.info(f"Currently showing frame from {time_sec} second(s). ")
        image_pil = Image.fromarray(st.session_state.frame_rgb)
        canvas_width = st.session_state.frame_rgb.shape[1]
        canvas_height = st.session_state.frame_rgb.shape[0]

        annotated_image = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="rgba(255, 0, 0, 1)",
            background_color="#eee",
            background_image=image_pil,
            update_streamlit=True,
            width=canvas_width,
            height=canvas_height,
            drawing_mode="rect",
            key=f"canvas_{time_sec}",
        )

        # Button to add bounding boxes from this frame to our dictionary
        if st.button("Add bounding boxes for this time"):
            if annotated_image.json_data is not None:
                shapes = annotated_image.json_data["objects"]
                bounding_boxes = []
                for shape in shapes:
                    if shape["type"] == "rect":
                        left = shape["left"]
                        top = shape["top"]
                        width = shape["width"]
                        height = shape["height"]
                        bounding_boxes.append((int(left), int(top), int(width), int(height)))

                if bounding_boxes:
                    if time_sec not in st.session_state.prompts_dict:
                        st.session_state.prompts_dict[time_sec] = []
                    # Append bounding boxes, in case user revisits same time
                    st.session_state.prompts_dict[time_sec].extend(bounding_boxes)
                    st.success(
                        f"Added {len(bounding_boxes)} bounding box(es) for second={time_sec} (total so far: {len(st.session_state.prompts_dict[time_sec])})"
                    )
                else:
                    st.warning("No bounding boxes drawn.")
            else:
                st.warning("No bounding boxes drawn.")

    st.write("Current prompts (time -> bounding boxes):")
    st.json(st.session_state.prompts_dict)


def plot():
    # Let's assume fps = 30
    fps = 30.0
    dt = 1.0 / fps

    cpts = np.array(st.session_state.center_points)  # shape (N,2)
    # Time array
    frames = len(cpts)
    if frames > 0:
        st.session_state.time = np.arange(frames) * dt
    else:
        st.session_state.time = np.array([])

    # Compute distances: Euclidean distance between consecutive points
    st.session_state.distances = np.zeros(frames)
    for i in range(1, frames):
        dx = cpts[i, 0] - cpts[i - 1, 0]
        dy = cpts[i, 1] - cpts[i - 1, 1]
        st.session_state.distances[i] = np.sqrt(dx * dx + dy * dy)

    # Cumulative distance
    st.session_state.cumulative_distance = np.cumsum(st.session_state.distances)

    # Velocity = distance/time (discrete derivative)
    # velocity here is instantaneous speed per frame
    st.session_state.velocity = np.zeros(frames)
    st.session_state.velocity[1:] = st.session_state.distances[1:] / dt

    # Acceleration = derivative of velocity
    st.session_state.acceleration = np.zeros(frames)
    st.session_state.acceleration[1:] = (st.session_state.velocity[1:] - st.session_state.velocity[:-1]) / dt

    # Save video
    st.success("Processing complete! Watch the output video below:")
    st.video(st.session_state.video_output_path, format="video/mp4")

    # Cumulative Distance Plot
    st.subheader("Cumulative Distance")
    st.session_state.cm_f = Figure(figsize=(4, 2))
    ax1 = st.session_state.cm_f.add_subplot(111)
    ax1.plot(st.session_state.time, st.session_state.cumulative_distance, color="blue", linewidth=0.5)
    ax1.set_xlabel("Time (s)", fontsize=6)
    ax1.set_ylabel("Distance (pixels)", fontsize=6)
    ax1.tick_params(axis='both', which='major', labelsize=6)
    ax1.set_xlim(left=0)  # Start X-axis at 0
    ax1.set_ylim(bottom=0)
    st.pyplot(st.session_state.cm_f)

    # Velocity Plot
    st.subheader("Velocity")
    st.session_state.cm_v = Figure(figsize=(4, 2))
    axv = st.session_state.cm_v.add_subplot(111)
    axv.plot(st.session_state.time, st.session_state.velocity, color="green", linewidth=0.5)
    axv.set_xlabel("Time (s)", fontsize=6)
    axv.set_ylabel("Velocity (pixels/s)", fontsize=6)
    axv.tick_params(axis='both', which='major', labelsize=6)
    axv.set_xlim(left=0)  # Start X-axis at 0
    axv.set_ylim(bottom=0)
    st.pyplot(st.session_state.cm_v)

    # Acceleration Plot
    st.subheader("Acceleration")
    st.session_state.cm_a = Figure(figsize=(4, 2))
    ax2 = st.session_state.cm_a.add_subplot(111)
    ax2.plot(st.session_state.time, st.session_state.acceleration, color="orange", linewidth=0.5)
    ax2.set_xlabel("Time (s)", fontsize=6)
    ax2.set_ylabel("Acceleration (pixels/s²)", fontsize=6)
    ax2.tick_params(axis='both', which='major', labelsize=6)
    ax2.set_xlim(left=0)
    st.pyplot(st.session_state.cm_a)


def prepare_save():
    with tempfile.TemporaryDirectory() as tmp_dir:
        results_folder = os.path.join(tmp_dir, "results")
        os.makedirs(results_folder, exist_ok=True)

        # Save CSV of center points
        center_points_df = pd.DataFrame(st.session_state.center_points, columns=["x", "y"])
        center_points_csv = os.path.join(results_folder, "center_points.csv")
        center_points_df.to_csv(center_points_csv, index=False)

        # Save CSV of distance, velocity, acceleration
        dva_df = pd.DataFrame({
            "time_s": st.session_state.time,
            "cumulative_distance_pixels": st.session_state.cumulative_distance,
            "velocity_pixels_per_s": st.session_state.velocity,
            "acceleration_pixels_per_s2": st.session_state.acceleration
        })
        dva_csv = os.path.join(results_folder, "distance_velocity_acceleration.csv")
        dva_df.to_csv(dva_csv, index=False)

        # Save figures as PNG
        dist_png = os.path.join(results_folder, "cumulative_distance.png")
        st.session_state.cm_f.savefig(dist_png, dpi=300)
        vel_png = os.path.join(results_folder, "velocity.png")
        st.session_state.cm_v.savefig(vel_png, dpi=300)
        acc_png = os.path.join(results_folder, "acceleration.png")
        st.session_state.cm_a.savefig(acc_png, dpi=300)

        # Also save the video
        video_dest = os.path.join(results_folder, "Quality.mp4")
        if os.path.exists(st.session_state.video_output_path):
            # Copy the video
            with open(st.session_state.video_output_path, "rb") as src:
                with open(video_dest, "wb") as dst:
                    dst.write(src.read())

        # Create an in-memory zip of the results folder
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Walk through the results_folder and add files
            for root, dirs, files in os.walk(results_folder):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, results_folder)
                    zf.write(full_path, rel_path)

        st.session_state.results_zip = zip_buffer.getvalue()


def process_video_and_analyze():
    st.subheader("Processing Video with Multiple Prompts")

    if st.session_state.video_output_path is None:
        if st.session_state.load_frames:
            with st.spinner("Saving frames ..."):
                save_video_frames_as_images(st.session_state.temp_video_path, st.session_state.output_dir)
                st.session_state.clean_folder = False
                st.session_state.load_frames = False

        with st.spinner("Processing frames ..."):
            list_im = sorted(glob.glob(os.path.join(st.session_state.output_dir, '*')))
            l_chunks = chunk_sequence(list_im, 200)

            # Prepare video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30
            # Suppose you know final frame size. Otherwise read the first frame to get size
            sample_im = cv2.imread(list_im[0])
            frame_height, frame_width = sample_im.shape[:2]
            frame_size = (frame_width, frame_height)
      
            video_output_path = "Quality.mp4"
            out = cv2.VideoWriter(video_output_path, fourcc, fps, frame_size)

            chunk_size = 200
            ann_obj_id = 1
            carried_bbox = None

            # List to store center points for further analysis
            center_points = []

            for chunk_idx, chunks in enumerate(tqdm.tqdm(l_chunks)):
                chunk_start_idx = chunk_idx * chunk_size
                chunk_end_idx = chunk_start_idx + len(chunks) - 1

                inference_state = predictor.init_state(video=chunks)
                predictor.reset_state(inference_state)

                # Possibly carry forward bounding box from last chunk's last frame
                first_frame_global_idx = chunk_start_idx
                time_sec_for_first_frame = first_frame_global_idx / fps

                has_user_prompt_for_first_frame = False
                if time_sec_for_first_frame in st.session_state.prompts_dict:
                    if len(st.session_state.prompts_dict[time_sec_for_first_frame]) > 0:
                        has_user_prompt_for_first_frame = True

                # If no user prompt for the chunk's first frame, use carried_bbox
                if (carried_bbox is not None) and (not has_user_prompt_for_first_frame):
                    box_prompt = np.array(carried_bbox)
                    _, _, _ = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=ann_obj_id,
                        box=box_prompt,
                    )

                # Add bounding boxes for frames in this chunk
                for time_sec, boxes in st.session_state.prompts_dict.items():
                    global_frame_idx = int(time_sec * fps)
                    if chunk_start_idx <= global_frame_idx <= chunk_end_idx:
                        ann_frame_idx = global_frame_idx - chunk_start_idx
                        for (x, y, w, h) in boxes:
                            x_min, y_min = x, y
                            x_max, y_max = x + w, y + h
                            box_prompt = np.array([x_min, y_min, x_max, y_max])
                            _, _, _ = predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=ann_frame_idx,
                                obj_id=ann_obj_id,
                                box=box_prompt
                            )

                last_bbox = None

                for out_sam, frame_path in zip(predictor.propagate_in_video(inference_state), chunks):
                    out_masks_logits = out_sam[2]
                    im = cv2.imread(frame_path)
                    overlay = im.copy()

                    for out_mask_logits in out_masks_logits:
                        boolean_mask = out_mask_logits > 0
                        integer_mask = boolean_mask.int().squeeze().cpu().numpy()

                        # Calculate the center of the mask
                        integer_mask_type_casted = integer_mask.astype(np.uint8)
                        #integer_mask_type_casted = keep_largest_connected_component(integer_mask_type_casted)
                        center_x, center_y = calculate_center(integer_mask_type_casted)

                        # Append the center point for analysis if valid
                        if center_x is not None and center_y is not None:
                            center_points.append((center_x, center_y))

                        # Mask overlay
                        colored_mask = np.zeros_like(im)
                        colored_mask[:, :, 1] = integer_mask * 255
                        overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

                        # BBox
                        try:
                            bbox = get_bbox(integer_mask)
                            cv2.rectangle(overlay, (bbox[0], bbox[1]),
                                          (bbox[2], bbox[3]), (0, 0, 256), 2)

                            last_bbox = bbox

                            # Draw the center point on the overlay if valid
                            if center_x is not None and center_y is not None:
                                cv2.circle(overlay, (center_x, center_y), 5, (0, 0, 255), -1)

                            # Draw trailing points (last 10)
                            for i, (tx, ty) in enumerate(center_points[-10:]):
                                alpha = (i + 1) / 10  # Gradual transparency
                                color = (0, int(255 * alpha), int(255 * (1 - alpha)))  # Fading green
                                cv2.circle(overlay, (tx, ty), 3, color, -1)
                        except:
                            pass

                    out.write(overlay)

                if last_bbox is not None:
                    carried_bbox = last_bbox
                else:
                    carried_bbox = None

            out.release()
            st.session_state.center_points = center_points
            st.session_state.video_output_path = video_output_path

            input_path = Path(st.session_state.video_output_path).resolve()
            output_path = Path("output_h264.mp4").resolve()

            if output_path.exists():
                output_path.unlink()  # Removes the file
                print(f"{output_path} has been removed.")
            else:
                print(f"{output_path} does not exist.")

            # Convert to H.264 codec
            ffmpeg_command = [
                "ffmpeg", "-y",  # -y to overwrite without asking
                "-i", str(input_path),  # Input file
                "-c:v", "libx264",  # Use H.264 codec
                "-preset", "fast",  # Speed vs compression trade-off
                "-crf", "23",  # Quality factor (lower is better)
                str(output_path)
            ]

            # Run the ffmpeg command
            subprocess.run(ffmpeg_command)

            # Update session state to new video path
            st.session_state.video_output_path = str(output_path)

    plot()

    # Option to add more bounding boxes
    st.info("Need more refinements or additional bounding boxes?")
    if st.button("Add More Bounding Boxes"):
        # We'll proceed to a new step (Step 5) that re-uses the same annotation UI
        st.session_state.step = 5
        st.session_state.video_output_path = None
        st.rerun()

    # Or we can continue to the download step
    if st.button("Go to Download Step"):
        st.session_state.step = 4
        prepare_save()
        st.rerun()


def main_app():
    st.title("Pig Video Analysis")

    # Delete files if needed (clean up)
    if st.session_state.clean_folder:
        delete_images_from_folder(st.session_state.output_dir)

    # Step 1: Upload the video if not done yet
    if st.session_state.step == 1:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_file is not None:
            # Write the uploaded video to a temporary file
            st.session_state.temp_video_path = "temp_video.mp4"
            with open(st.session_state.temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract the first frame
            cap = cv2.VideoCapture(st.session_state.temp_video_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                st.session_state.step = 2
                st.rerun()
            else:
                st.error("Could not read the video file. Make sure it's a supported format.")
        else:
            st.info("Please upload a video file to continue.")

    # Step 2: Add bounding boxes
    if st.session_state.step == 2:
        step_2_annotation()
        if st.button("Done with annotation. Proceed to Analysis."):
            if len(st.session_state.prompts_dict) == 0:
                st.warning("You haven't added any bounding boxes yet!")
            else:
                st.session_state.step = 3
                st.rerun()

    # Step 3: Process video with current bounding boxes prompts
    if st.session_state.step == 3:
        process_video_and_analyze()

    # Step 5: Download the results
    if st.session_state.step == 4:
        st.subheader("Download Results")
        st.write("Click the button below to download all results (CSV files, plots, and video) as a zip file:")

        # Download results zip
        if "results_zip" in st.session_state and st.session_state.results_zip is not None:
            st.download_button(
                label="Download Results Zip",
                data=st.session_state.results_zip,
                file_name="results.zip",
                mime="application/zip",
            )
        else:
            st.warning("No results found. Please go back and run the analysis first.")

    # Step 5: Additional annotation (same UI as Step 2, but re-run from Step 3 afterwards)
    elif st.session_state.step == 5:
        st.info("Add more bounding boxes or refine existing prompts (same UI).")
        step_2_annotation()
        if st.button("Done. Re-run Analysis with New Prompts"):
            # Clear old outputs so we re-process everything
            st.session_state.video_output_path = None
            st.session_state.center_points = None
            st.session_state.step = 3
            st.rerun()


# ---------------------
# Main Execution
# ---------------------
if not st.session_state.logged_in:
    login_screen()
else:
    main_app()


