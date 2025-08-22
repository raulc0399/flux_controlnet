import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # Input configuration
    repo_name_text = mo.ui.text(value="raulc/open_pose_controlnet_dataset_small", label="Repo Name")
    dataset_text = mo.ui.text(value="train.parquet", label="Dataset File")
    images_directory_text = mo.ui.text(value="images", label="Images Directory")
    detect_resolution_slider = mo.ui.slider(start=512, stop=2048, value=1024, step=256, label="Detection Resolution")
    image_resolution_slider = mo.ui.slider(start=512, stop=2048, value=1024, step=256, label="Image Resolution")
    hand_and_face_checkbox = mo.ui.checkbox(value=True, label="Include Hand and Face Detection")
    
    process_button = mo.ui.run_button(label="Generate Pose Control Images")
    
    mo.vstack([
        mo.md("# Pose Control Image Generator"),
        mo.md("Generate OpenPose control images for ControlNet training"),
        repo_name_text,
        dataset_text,
        images_directory_text,
        detect_resolution_slider,
        image_resolution_slider,
        hand_and_face_checkbox,
        process_button
    ])
    return (
        dataset_text,
        detect_resolution_slider,
        hand_and_face_checkbox,
        image_resolution_slider,
        images_directory_text,
        mo,
        process_button,
        repo_name_text,
    )


@app.cell
def _(
    dataset_text,
    detect_resolution_slider,
    generate_pose_images,
    hand_and_face_checkbox,
    image_resolution_slider,
    images_directory_text,
    mo,
    process_button,
    repo_name_text,
):
    mo.stop(not process_button.value)

    generate_pose_images(
        repo_name=repo_name_text.value,
        dataset_file=dataset_text.value,
        images_directory=images_directory_text.value,
        detect_resolution=detect_resolution_slider.value,
        image_resolution=image_resolution_slider.value,
        hand_and_face=hand_and_face_checkbox.value
    )
    return


@app.cell
def _(OpenposeDetector, RemoteRepo, os, pd, process_image):
    def generate_pose_images(repo_name, dataset_file, images_directory,
                           detect_resolution=1024, image_resolution=1024, hand_and_face=True):
        """Main function to generate pose control images"""
        
        # Setup repository
        repo = RemoteRepo(repo_name)
        
        # Create a unique experiment branch name
        experiment_prefix = "pose-generation"
        branches = repo.branches()
        experiment_number = 0
        for branch in branches:
            if branch.name.startswith(experiment_prefix):
                experiment_number += 1
        branch_name = f"{experiment_prefix}-{experiment_number}"
        print(f"Creating branch: {branch_name}")
        repo.create_checkout_branch(branch_name)
        
        # Download dataset and images if they don't exist
        if not os.path.exists(dataset_file):
            print("Downloading dataset")
            repo.download(dataset_file)
            
        if not os.path.exists(images_directory):
            print("Downloading images")
            repo.download(images_directory)
        
        # Load the dataset
        df = pd.read_parquet(dataset_file)
        
        # Get list of images to process from the dataset
        files_to_process = []
        already_processed_files = []
        
        # Check which pose images already exist
        for index, row in df.iterrows():
            image_file = row['image'].replace('./', '')  # Remove ./ prefix
            conditioning_image_file = row['conditioning_image'].replace('./', '')  # Remove ./ prefix
            
            image_path = os.path.join(images_directory, os.path.basename(image_file))
            pose_path = os.path.join(images_directory, os.path.basename(conditioning_image_file))
            
            if os.path.exists(image_path) and not os.path.exists(pose_path):
                files_to_process.append(os.path.basename(image_file))
            elif os.path.exists(pose_path):
                already_processed_files.append(os.path.basename(conditioning_image_file))

        count_files = len(files_to_process)
        print(f"Processing {count_files} images")
        print(f"Already processed: {len(already_processed_files)} images")

        if count_files == 0:
            print("No new images to process!")
            return

        # Create OpenPose detector instance
        print("Loading OpenPose detector...")
        detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

        # Process images
        from tqdm import tqdm
        
        processed_count = 0
        processed_files = []
        
        with tqdm(total=count_files, desc="Processing images") as pbar:
            for file_name in files_to_process:
                result = process_image(
                    file_name,
                    images_directory,
                    detector,
                    detect_resolution,
                    image_resolution,
                    hand_and_face
                )
                if result:
                    processed_count += 1
                    processed_files.append(result)
                    
                pbar.set_description(f"With pose: {processed_count}")
                pbar.update(1)

        print(f"Total processed images: {processed_count}")
        
        # Upload processed files to repository
        if processed_files:
            print("Uploading processed images to repository...")
            for pose_file in processed_files:
                repo.add(pose_file, dst=images_directory)
            
            repo.commit(f"Generated {processed_count} pose control images")
            print(f"✅ Uploaded {processed_count} pose images to branch {branch_name}")
        
        print("Pose generation completed!")

    return (generate_pose_images,)


@app.cell
def _(Image, os):
    def process_image(file_name, images_directory, open_pose_instance,
                     detect_resolution=1024, image_resolution=1024, hand_and_face=True):
        """Process a single image to generate pose control image"""
        file_path = os.path.join(images_directory, file_name)
        output_file_name = f"{os.path.splitext(file_name)[0]}_pose.jpg"
        output_path = os.path.join(images_directory, output_file_name)

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
                return output_path

        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
        
        return None

    return (process_image,)




@app.cell
def _():
    # Import required libraries
    from controlnet_aux.processor import OpenposeDetector
    from PIL import Image
    from tqdm import tqdm
    from oxen import RemoteRepo
    import pandas as pd
    import os
    import gc

    return (
        Image,
        OpenposeDetector,
        RemoteRepo,
        gc,
        os,
        pd,
        tqdm,
    )


if __name__ == "__main__":
    app.run()
