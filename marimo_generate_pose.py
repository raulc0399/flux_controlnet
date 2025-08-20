import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # Input configuration
    input_folder_text = mo.ui.text(value="../persons-photo-concept-bucket-images-to-train", label="Input Images Folder")
    output_folder_text = mo.ui.text(value="pose", label="Output Folder Name (relative to input)")
    detect_resolution_slider = mo.ui.slider(start=512, stop=2048, value=1024, step=256, label="Detection Resolution")
    image_resolution_slider = mo.ui.slider(start=512, stop=2048, value=1024, step=256, label="Image Resolution")
    hand_and_face_checkbox = mo.ui.checkbox(value=True, label="Include Hand and Face Detection")
    
    process_button = mo.ui.run_button(label="Generate Pose Control Images")
    
    mo.vstack([
        mo.md("# Pose Control Image Generator"),
        mo.md("Generate OpenPose control images for ControlNet training"),
        input_folder_text,
        output_folder_text,
        detect_resolution_slider,
        image_resolution_slider,
        hand_and_face_checkbox,
        process_button
    ])
    return (
        detect_resolution_slider,
        hand_and_face_checkbox,
        image_resolution_slider,
        input_folder_text,
        mo,
        output_folder_text,
        process_button,
    )


@app.cell
def _(
    detect_resolution_slider,
    generate_pose_images,
    hand_and_face_checkbox,
    image_resolution_slider,
    input_folder_text,
    mo,
    output_folder_text,
    process_button,
):
    mo.stop(not process_button.value)

    generate_pose_images(
        input_folder=input_folder_text.value,
        output_folder_name=output_folder_text.value,
        detect_resolution=detect_resolution_slider.value,
        image_resolution=image_resolution_slider.value,
        hand_and_face=hand_and_face_checkbox.value
    )
    return


@app.cell
def _(OpenposeDetector, os, process_image):
    def generate_pose_images(input_folder, output_folder_name="pose", 
                           detect_resolution=1024, image_resolution=1024, hand_and_face=True):
        """Main function to generate pose control images"""
        
        # Create output folder path
        pose_folder = os.path.join(input_folder, output_folder_name)
        
        if not os.path.exists(pose_folder):
            os.makedirs(pose_folder)
            print(f"Created output directory: {pose_folder}")
        
        # Get list of files to process
        files_to_process = []
        files_in_directory = os.listdir(input_folder)
        already_processed_files = os.listdir(pose_folder) if os.path.exists(pose_folder) else []

        for file_name in files_in_directory:
            input_file_path = os.path.join(input_folder, file_name)
            pose_file_name = f"{os.path.splitext(file_name)[0]}_pose.jpg"

            if (os.path.isfile(input_file_path) and 
                input_file_path.lower().endswith(('.jpg', '.jpeg', '.png')) and 
                pose_file_name not in already_processed_files):
                files_to_process.append(file_name)

        count_files = len(files_to_process)
        print(f"Processing {count_files} images")

        if count_files == 0:
            print("No new images to process!")
            return

        # Create OpenPose detector instance
        print("Loading OpenPose detector...")
        detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

        # Process images
        from tqdm import tqdm
        
        processed_count = 0
        with tqdm(total=count_files, desc="Processing images") as pbar:
            for file_name in files_to_process:
                processed_count += process_image(
                    file_name,
                    input_folder,
                    pose_folder,
                    detector,
                    detect_resolution,
                    image_resolution,
                    hand_and_face
                )
                pbar.set_description(f"With pose: {processed_count}")
                pbar.update(1)

        print(f"Total processed images: {processed_count}")
        print("Pose generation completed!")

    return (generate_pose_images,)


@app.cell
def _(Image, os):
    def process_image(file_name, folder, pose_folder, open_pose_instance,
                     detect_resolution=1024, image_resolution=1024, hand_and_face=True):
        """Process a single image to generate pose control image"""
        file_path = os.path.join(folder, file_name)
        output_file_name = f"{os.path.splitext(file_name)[0]}_pose.jpg"
        output_path = os.path.join(pose_folder, output_file_name)

        try:
            img = Image.open(file_path)
            processed_image_open_pose = open_pose_instance(
                img, 
                hand_and_face=hand_and_face, 
                detect_resolution=detect_resolution, 
                image_resolution=image_resolution
            )
            
            if processed_image_open_pose is not None:
                processed_image_open_pose.save(output_path)
                # Copy original image to pose folder as well
                os.system(f"cp {file_path} {os.path.join(pose_folder, file_name)}")
                return 1

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
        
        return 0

    return (process_image,)




@app.cell
def _():
    # Import required libraries
    from controlnet_aux.processor import OpenposeDetector
    from PIL import Image
    from tqdm import tqdm
    import os
    import gc

    return (
        Image,
        OpenposeDetector,
        gc,
        os,
        tqdm,
    )


if __name__ == "__main__":
    app.run()
