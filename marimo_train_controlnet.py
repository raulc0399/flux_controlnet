# copy of https://www.oxen.ai/ox/Fine-Tune-FLUX/file/main/train.py

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    # https://huggingface.co/black-forest-labs/FLUX.1-dev
    model_name_text = mo.ui.text(value="black-forest-labs/FLUX.1-dev", label="Model Name")
    repo_name_text = mo.ui.text(value="raulc/open_pose_controlnet_dataset_small", label="Repo Name")
    dataset_text = mo.ui.text(value="train.parquet", label="Dataset File")
    images_directory_text = mo.ui.text(value="images", label="Images Directory")
    
    # ControlNet configuration
    use_controlnet_lora_checkbox = mo.ui.checkbox(value=True, label="Use LoRA for ControlNet")
    num_double_layers_slider = mo.ui.slider(start=1, stop=8, value=4, step=1, label="Number of Double Layers")
    num_single_layers_slider = mo.ui.slider(start=1, stop=8, value=4, step=1, label="Number of Single Layers")
    controlnet_conditioning_scale_slider = mo.ui.slider(start=0.1, stop=2.0, value=1.0, step=0.1, label="ControlNet Conditioning Scale")
    
    hf_api_key_text = mo.ui.text(kind="password", label="Hugging Face API Key")
    button = mo.ui.run_button(label="Train ControlNet Model")
    mo.vstack([
        model_name_text,
        repo_name_text,
        dataset_text,
        images_directory_text,
        mo.md("**ControlNet Configuration**"),
        use_controlnet_lora_checkbox,
        num_double_layers_slider,
        num_single_layers_slider,
        controlnet_conditioning_scale_slider,
        hf_api_key_text,
        button
    ])
    return (
        button,
        controlnet_conditioning_scale_slider,
        dataset_text,
        hf_api_key_text,
        images_directory_text,
        mo,
        model_name_text,
        num_double_layers_slider,
        num_single_layers_slider,
        repo_name_text,
        use_controlnet_lora_checkbox,
    )


@app.cell
def _(
    button,
    controlnet_conditioning_scale_slider,
    dataset_text,
    hf_api_key_text,
    images_directory_text,
    mo,
    model_name_text,
    num_double_layers_slider,
    num_single_layers_slider,
    repo_name_text,
    train,
    use_controlnet_lora_checkbox,
):
    mo.stop(not button.value)

    train(
        model_name_text.value,
        repo_name_text.value,
        dataset_text.value,
        images_directory_text.value,
        hf_api_key_text.value,
        use_controlnet_lora_checkbox.value,
        num_double_layers_slider.value,
        num_single_layers_slider.value,
        controlnet_conditioning_scale_slider.value
    )
    return


@app.cell
def _(app, mo):
    if mo.app_meta().mode == "script":
        # Run our CLI app
        app()
    return


@app.cell
def _(
    DataLoader,
    F,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetDataset,
    RemoteRepo,
    bnb,
    datetime,
    flush_memory,
    generate_samples,
    json,
    load_models,
    os,
    rearrange,
    torch,
    write_and_save_results,
):
    import typer
    import time
    from huggingface_hub import login as hf_login

    # Create the CLI app
    app = typer.Typer()

    # Add entry point to your CLI app
    @app.command()
    def train(model_name, repo_name, dataset_file, images_directory, hf_api_key, 
              use_controlnet_lora, num_double_layers, num_single_layers, controlnet_conditioning_scale):
        # Must login to Hugging Face to download the weights
        hf_login(hf_api_key)

        # Brain Float 16
        dtype = torch.bfloat16
        device = torch.device("cuda")
        lora_rank = 16
        lora_alpha = 16
        models = load_models(model_name, lora_rank, lora_alpha, use_controlnet_lora=use_controlnet_lora,
                           num_double_layers=num_double_layers, num_single_layers=num_single_layers)

        config = {
            # Training settings
            "trigger_phrase": "Finn the dog",
            "dataset_repo": repo_name,
            "dataset_path": dataset_file,
            "images_path": images_directory,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "steps": 2000,
            "learning_rate": 1e-4,
            "optimizer": "adamw8bit",
            "noise_scheduler": "flowmatch",

            # LoRA settings - Updated to match YAML config
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "use_controlnet_lora": use_controlnet_lora,
            
            # ControlNet settings
            "num_double_layers": num_double_layers,
            "num_single_layers": num_single_layers,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,

            # Save settings
            "save_every": 200,
            "sample_every": 200,
            "max_step_saves": 4,
            "save_dtype": "float16",

            # Sample settings - Updated to match YAML config
            "sample_width": 1024,
            "sample_height": 1024,
            "guidance_scale": 3.5,
            "sample_steps": 30,
            "sample_prompts": [
                "[trigger] playing chess",
                "[trigger] holding a coffee cup",
                "[trigger] DJing at a night club",
                "[trigger] wearing a blue beanie",
                "[trigger] flying a kite",
                "[trigger] fixing an upside down bicycle",
            ]
        }

        dataset = FluxControlNetDataset(
            config["dataset_repo"],
            config["dataset_path"],
            config["images_path"],
            config["control_images_path"],
            trigger_phrase=config["trigger_phrase"]
        )

        dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=True
        )

        # We loaded the model previously, and saved all the components in a tuple
        (transformer, vae, clip_encoder, t5_encoder, clip_tokenizer, t5_tokenizer, flux_controlnet) = models

        # Make sure the transformer's parameters are NOT trainable
        transformer.eval()
        # ControlNet should also be trainable
        flux_controlnet.train()

        print("Loading Noise Scheduler")
        # Stable Diffusion 3 https://arxiv.org/abs/2403.03206
        # FlowMatchEulerDiscreteScheduler is based on the flow-matching sampling introduced in Stable Diffusion 3.
        # Dynamic shifting works well for high resolution images, where we want to add a lot of noise at the start
        flux_scheduler_config = {
            "shift": 3.0,
            "use_dynamic_shifting": True,
            "base_shift": 0.5,
            "max_shift": 1.15
        }
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name,
            subfolder="scheduler",
            torch_dtype=dtype
        )

        # Update scheduler config with Flux-specific parameters
        for key, value in flux_scheduler_config.items():
            if hasattr(noise_scheduler.config, key):
                setattr(noise_scheduler.config, key, value)

        print(f"Scheduler config updated: shift={noise_scheduler.config.shift}, use_dynamic_shifting={noise_scheduler.config.use_dynamic_shifting}")

        print("Setting up optimizer")
        # Only train ControlNet parameters, keep transformer frozen
        optimizer = bnb.optim.AdamW8bit(
            flux_controlnet.parameters(),
            lr=config["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )

        # Set up learning rate scheduler, in this case, constant
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        # Upload results to Oxen.ai
        save_repo_name = repo_name
        # RemoteRepo is from the oxenai python lib
        repo = RemoteRepo(save_repo_name)

        # Create a unique experiment branch name
        experiment_prefix = f"fine-tune"
        branches = repo.branches()
        experiment_number = 0
        for branch in branches:
            if branch.name.startswith(experiment_prefix):
                experiment_number += 1
        branch_name = f"{experiment_prefix}-{config['lora_rank']}-{config['lora_alpha']}-{experiment_number}"
        print(f"Experiment name: {branch_name}")
        repo.create_checkout_branch(branch_name)

        # Create the output dir
        output_dir = os.path.join("output", branch_name)
        os.makedirs(output_dir, exist_ok=True)

        # Write the config file
        config_file = os.path.join(output_dir, 'training_config.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(config))
        repo.add(config_file, dst=output_dir)

        # Save logs for debugging
        global_step = 0
        training_logs = []
        image_logs = []
        readme_lines = []

        # Save images to a README that we can look at during training
        readme_lines.append(f"# Flux Fine-Tune\n\n")
        readme_lines.append(f"Automatically generated during training `{model_name}` and saving to branch `{branch_name}`.\n\nBelow are some samples from the training run\n\n")

        # Training Loop
        while global_step < config["steps"]:
            for batch in dataloader:
                if global_step >= config["steps"]:
                    break

                # Autocast will convert to dtype=bfloat16 and ensure conformity
                with torch.amp.autocast('cuda', dtype=dtype):
                    # Grab the images and control images from the batch
                    images = batch['image'].to(device, dtype=dtype)
                    control_images = batch['control_image'].to(device, dtype=dtype)

                    # Encode the images to the latent space
                    latents = vae.encode(images).latent_dist.sample()
                    
                    # Encode the control images to the latent space
                    control_latents = vae.encode(control_images).latent_dist.sample()

                    # Scale and shift the latents to help with training stability.
                    scaling_factor = vae.config.scaling_factor
                    shift_factor = vae.config.shift_factor
                    # When encoding: (x - shift) * scale
                    latents = (latents - shift_factor) * scaling_factor
                    control_latents = (control_latents - shift_factor) * scaling_factor

                    # CLIP tokenization and encoding
                    clip_inputs = clip_tokenizer(
                        batch['caption'],
                        max_length=clip_tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )

                    clip_outputs = clip_encoder(
                        input_ids=clip_inputs.input_ids.to(device),
                        attention_mask=clip_inputs.attention_mask.to(device),
                    )
                    pooled_embeds = clip_outputs.pooler_output

                    # T5 tokenization and encoding
                    t5_inputs = t5_tokenizer(
                        batch['caption'],
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    )

                    t5_outputs = t5_encoder(
                        input_ids=t5_inputs.input_ids.to(device),
                        attention_mask=t5_inputs.attention_mask.to(device),
                    )
                    prompt_embeds = t5_outputs.last_hidden_state

                    # Sample noise with the same shape as the latents
                    noise = torch.randn_like(latents)

                    # Sample from 1000 timesteps
                    num_timesteps = 1000
                    t = torch.sigmoid(torch.randn((num_timesteps,), device=device))

                    # Scale and reverse the values to go from 1000 to 0
                    timesteps = ((1 - t) * num_timesteps)

                    # Sort the timesteps in descending order
                    timesteps, _ = torch.sort(timesteps, descending=True)
                    timesteps = timesteps.to(device=device)

                    # Use uniform timestep sampling
                    # Sample timestep indices uniformly - use actual length of timesteps array
                    min_noise_steps = 0
                    max_noise_steps = num_timesteps
                    timestep_indices = torch.randint(
                        min_noise_steps,  # min_idx for flowmatch
                        max_noise_steps - 1,  # max_idx (exclusive upper bound)
                        (config['batch_size'],),
                        device=device
                    ).long()

                    # Convert indices to actual timesteps using scheduler's timesteps array
                    timesteps = timesteps[timestep_indices]

                    # Get the percentage of the timestep
                    t_01 = (timesteps / num_timesteps).to(latents.device)

                    # Forward ODE for Rectified Flow
                    # zt = (1 − t)x0 + tϵ
                    noisy_latents = (1.0 - t_01) * latents + t_01 * noise

                    # Pack latents for FLUX (similar to FluxControlNetPipeline._pack_latents)
                    packed_latents = rearrange(
                        noisy_latents,
                        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                        ph=2, pw=2
                    )
                    
                    # Pack control latents
                    packed_control_latents = rearrange(
                        control_latents,
                        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                        ph=2, pw=2
                    )

                    def generate_position_ids_flux(batch_size, latent_height, latent_width, device):
                        """Generate position IDs for Flux transformer based on latent dimensions"""
                        # Position IDs for packed latents (2x2 packing reduces dimensions by half)
                        packed_h, packed_w = latent_height // 2, latent_width // 2
                        img_ids = torch.zeros(packed_h, packed_w, 3, device=device)
                        img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_h, device=device)[:, None]
                        img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_w, device=device)[None, :]
                        img_ids = rearrange(img_ids, "h w c -> (h w) c")

                        return img_ids

                    # Generate position IDs based on latent dimensions (not pixel dimensions)
                    img_ids = generate_position_ids_flux(config['batch_size'], latents.shape[2], latents.shape[3], device)
                    txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device)

                    # Guidance embedding
                    # I was getting blurry images if this was high during training
                    guidance_embedding_scale = 1.0
                    guidance = torch.tensor([guidance_embedding_scale], device=device, dtype=dtype)
                    guidance = guidance.expand(latents.shape[0])

                    # Forward pass - Use consistent timestep scaling
                    timestep_scaled = timesteps.float() / num_timesteps  # Consistent with t_01 scaling
                    
                    # ControlNet forward pass
                    controlnet_block_samples, controlnet_single_block_samples = flux_controlnet(
                        hidden_states=packed_latents,
                        controlnet_cond=packed_control_latents,
                        timestep=timestep_scaled,
                        guidance=guidance,
                        pooled_projections=pooled_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=txt_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )

                    # Transformer forward pass with ControlNet conditioning
                    noise_pred = transformer(
                        hidden_states=packed_latents,
                        timestep=timestep_scaled,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        txt_ids=txt_ids,
                        img_ids=img_ids,
                        guidance=guidance,
                        return_dict=False
                    )[0]

                    height, width = latents.shape[2], latents.shape[3]
                    noise_pred = rearrange(
                        noise_pred,
                        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                        h=height // 2,
                        w=width // 2,
                        ph=2, pw=2,
                        c=vae.config.latent_channels # Flux latent channels
                    )

                    # Flow matching loss target - rectified flow formulation
                    target = (noise - latents).detach()

                    # Calculate loss without reduction first for timestep weighting
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")

                    # Reduce to scalar
                    loss = loss.mean()

                    # Compute the gradients
                    loss.backward()

                    # Clip global L2 norm of all gradients to ≤1.0 to prevent exploding updates
                    # Helpful when training in bfloat16. It also plays well with AdamW.
                    torch.nn.utils.clip_grad_norm_(flux_controlnet.parameters(), 1.0)

                    # Optimizer step to update the weights
                    optimizer.step()

                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Collect training logs as we go
                    t_log = {
                        "step": global_step,
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    training_logs.append(t_log)
                    print(t_log)

                    # Generate samples (skip early steps to avoid dtype issues)
                    if global_step % config["sample_every"] == 0:
                        image_paths = generate_samples(
                            config, transformer, vae, clip_encoder, t5_encoder,
                            clip_tokenizer, t5_tokenizer, noise_scheduler, flux_controlnet,
                            config["sample_prompts"], # Generate samples from the config prompt list
                            output_dir, global_step, device, dtype
                        )
                        readme_lines.append(f"## Sample Images {global_step}\n\n")
                        for image_path in image_paths:
                            image_logs.append({
                                "step": global_step,
                                "image": image_path,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            url = f"https://hub.oxen.ai/api/repos/{save_repo_name}/file/{branch_name}/{image_path}"
                            readme_lines.append(f'<a href="{url}"><img src="{url}" width="256" height="256" /></a>\n')

                        with open(os.path.join(output_dir, "README.md"), "w") as f:
                            for line in readme_lines:
                                f.write(line + "\n")

                    # Save checkpoint
                    if global_step % config["save_every"] == 0:
                        save_path = f"flux_controlnet_step_{global_step}.safetensors"

                        # Save model weights
                        write_and_save_results(flux_controlnet, repo, output_dir, save_path, training_logs, image_logs)

                    global_step += 1


                    # Was getting OOM errors after sampling
                    flush_memory()

        # Final save
        final_save_path = os.path.join(output_dir, "flux_controlnet_final.safetensors")
        write_and_save_results(flux_controlnet, repo, output_dir, "flux_controlnet_final.safetensors", training_logs, image_logs)

        print("Training completed!")
    return app, train


@app.cell
def _(FluxControlNetPipeline, flush_memory, os, torch):
    @torch.no_grad()
    def generate_samples(config, transformer, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                        scheduler, flux_controlnet, prompts, output_dir, step, device, dtype):
        """Generate sample images during training"""

        transformer.eval()
        flux_controlnet.eval()

        # This matches ai-toolkit's approach better
        sample_dtype = torch.float32 if dtype == torch.bfloat16 else dtype

        # Ensure all models are on the same device with consistent dtype
        vae = vae.to(device=device, dtype=sample_dtype)
        text_encoder = text_encoder.to(device=device, dtype=sample_dtype)
        text_encoder_2 = text_encoder_2.to(device=device, dtype=sample_dtype)

        # Create ControlNet pipeline for sampling
        pipeline = FluxControlNetPipeline(
            transformer=transformer,
            controlnet=flux_controlnet,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler
        )

        # Move pipeline to device with compatible dtype
        pipeline = pipeline.to(device=device, dtype=sample_dtype)

        # Load a control image for sampling (use first one from dataset)
        from PIL import Image as PILImage
        try:
            control_image_path = os.path.join(config["control_images_path"], os.listdir(config["control_images_path"])[0])
            control_image = PILImage.open(control_image_path).convert('RGB')
            control_image = control_image.resize((config["sample_width"], config["sample_height"]), PILImage.LANCZOS)
        except:
            # If no control image available, create a simple black image
            control_image = PILImage.new('RGB', (config["sample_width"], config["sample_height"]), (0, 0, 0))
        
        image_paths = []
        try:
            for i, prompt in enumerate(prompts):
                # Replace trigger placeholder if present
                if "[trigger]" in prompt and config["trigger_phrase"]:
                    prompt = prompt.replace("[trigger]", config["trigger_phrase"])
                elif config["trigger_phrase"] and config["trigger_phrase"] not in prompt:
                    # Add trigger word to prompt if configured (only if no [trigger] placeholder)
                    prompt = f"{config['trigger_phrase']}{prompt}"

                image = pipeline(
                    prompt=prompt,
                    control_image=control_image,
                    width=config["sample_width"],
                    height=config["sample_height"],
                    num_inference_steps=config["sample_steps"],
                    guidance_scale=config["guidance_scale"],
                    controlnet_conditioning_scale=config["controlnet_conditioning_scale"],
                    generator=torch.Generator(device=device).manual_seed(42 + i),
                ).images[0]

                # Save image
                sample_dir = os.path.join(output_dir, "samples")
                os.makedirs(sample_dir, exist_ok=True)

                # Create the filename from the prompt
                prompt_prefix = "_".join(prompt.split(" ")[-8:])
                sample_path = os.path.join(sample_dir, f"step_{step}_sample_{i}_{prompt_prefix}.png")
                image.save(sample_path)

                print(f"Saved sample: {sample_path}")

                image_paths.append(sample_path)

        except Exception as e:
            print(f"Error generating samples: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")

        finally:
            # transformer.train()
            flux_controlnet.train()

            # Restore original dtypes
            vae = vae.to(device=device, dtype=dtype)
            text_encoder = text_encoder.to(device=device, dtype=dtype)
            text_encoder_2 = text_encoder_2.to(device=device, dtype=dtype)

            # Was getting OOM errors
            flush_memory()

        return image_paths

    return (generate_samples,)


@app.cell
def _(gc, torch):
    def flush_memory():
        """Flush GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return (flush_memory,)


@app.cell
def _(e, json, os, save_lora_weights, torch):
    def write_and_save_results(model, repo, output_dir, model_name, training_logs, image_logs):
        try:
            # Final save
            final_save_path = os.path.join(output_dir, model_name)
            save_lora_weights(model, final_save_path, torch.float16)

            # Save training logs
            training_logs_file = os.path.join(output_dir, "training_logs.jsonl")
            with open(os.path.join(output_dir, "training_logs.jsonl"), "w") as f:
                for log in training_logs:
                    f.write(json.dumps(log) + "\n")

            # Save image logs
            image_logs_file = os.path.join(output_dir, "image_logs.jsonl")
            with open(image_logs_file, "w") as f:
                for log in image_logs:
                    f.write(json.dumps(log) + "\n")

            repo.add(final_save_path, dst=output_dir)
            repo.add(training_logs_file, dst=output_dir)
            repo.add(image_logs_file, dst=output_dir)

            readme_file = os.path.join(output_dir, "README.md")
            repo.add(readme_file, dst=output_dir)

            samples_dir = os.path.join(output_dir, "samples")
            repo.add(samples_dir, dst=samples_dir)
            repo.commit(f"Saving final model {output_dir}")
            print("✅ Uploaded checkpoint")
        except e:
            print(f"😢 Could not save weights {e}")

    return (write_and_save_results,)


@app.cell
def _(Dataset, Image, RemoteRepo, os, pd, random, transforms):
    class FluxControlNetDataset(Dataset):
        """Dataset for loading images, captions, and control images for Flux ControlNet training"""

        def __init__(self, dataset_repo, dataset_path, images_path, control_images_path, resolutions=[512, 768, 1024], trigger_phrase=None):
            self.repo = RemoteRepo(dataset_repo)
            self.resolutions = resolutions
            self.trigger_phrase = trigger_phrase
            self.images_path = images_path
            self.control_images_path = control_images_path

            if not os.path.exists(images_path):
                print("Downloading images")
                self.repo.download(images_path)
            
            if not os.path.exists(dataset_path):
                print("Downloading dataset")
                self.repo.download(dataset_path)

            # Load the dataset
            df = pd.read_parquet(dataset_path)

            # Read all images, captions, and control images
            self.image_files = []
            self.control_image_files = []
            self.captions = []
            for index, row in df.iterrows():
                self.image_files.append(row['image'])
                self.captions.append(row['action'])
                # Assume control image has same name as regular image or use 'conditioning_image' column if available
                if 'conditioning_image' in row:
                    self.control_image_files.append(row['conditioning_image'])
                else:
                    # Use same filename for control image (assumes same naming)
                    self.control_image_files.append(row['image'])

            # Setup transforms
            # You could add cropping and rotating here if you wanted
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            print(f"Found {len(self.image_files)} images in {dataset_path}")

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            # Load image and control image
            image_path = self.image_files[idx]
            control_image_path = self.control_image_files[idx]
            caption = self.captions[idx]
            image = Image.open(os.path.join(self.images_path, image_path)).convert('RGB')
            control_image = Image.open(os.path.join(self.control_images_path, control_image_path)).convert('RGB')

            # Add trigger word if specified and not already present
            if self.trigger_phrase and self.trigger_phrase not in caption:
                caption = f"{self.trigger_phrase}{caption}" if caption else self.trigger_phrase

            # Random resolution for multi-aspect training
            target_res = random.choice(self.resolutions)

            # Resize image maintaining aspect ratio
            width, height = image.size
            if width > height:
                new_width = target_res
                new_height = int(height * target_res / width)
            else:
                new_height = target_res
                new_width = int(width * target_res / height)

            # Make dimensions divisible by 16 (Flux requirement)
            new_width = (new_width // 16) * 16
            new_height = (new_height // 16) * 16

            image = image.resize((new_width, new_height), Image.LANCZOS)
            control_image = control_image.resize((new_width, new_height), Image.LANCZOS)
            
            image = self.transform(image)
            control_image = self.transform(control_image)

            return {
                'image': image,
                'control_image': control_image,
                'caption': caption,
                'width': new_width,
                'height': new_height
            }
    return (FluxControlNetDataset,)


@app.cell
def _(
    AutoencoderKL,
    CLIPTextModel,
    CLIPTokenizer,
    FluxControlNetModel,
    FluxTransformer2DModel,
    LoraConfig,
    T5EncoderModel,
    T5TokenizerFast,
    get_peft_model,
    torch,
):
    def load_models(model_name, lora_rank=16, lora_alpha=16, dtype=torch.bfloat16, device="cuda", 
                   use_controlnet_lora=True, num_double_layers=4, num_single_layers=4):
        # Transfor all the models to the GPU device at the end
        device = torch.device(device)

        # Load transformer
        print("Loading FluxTransformer2DModel")
        transformer = FluxTransformer2DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            torch_dtype=dtype
        )
        # For more efficient memory usage during training
        # transformer.enable_gradient_checkpointing()
        transformer.eval()

        # Apply LoRA
        print("Applying LoRA FluxTransformer2DModel")
        # Target modules for LoRA (Flux transformer specific modules)
        target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",  # Attention layers
            "ff.net.0.proj", "ff.net.2",  # MLP layers
            "proj_out"  # Output projection
        ]
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        transformer = get_peft_model(transformer, lora_config)
        transformer.print_trainable_parameters()

        # Load VAE
        print("Loading AutoencoderKL")
        vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=dtype
        )
        vae.eval()

        # Load text encoders
        print("Loading CLIPTextModel")
        text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            torch_dtype=dtype
        )
        text_encoder.eval()

        print("Loading T5EncoderModel")
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_name,
            subfolder="text_encoder_2",
            torch_dtype=dtype
        )
        text_encoder_2.eval()

        # Load tokenizers
        print("Loading CLIPTokenizer")
        tokenizer = CLIPTokenizer.from_pretrained(
            model_name,
            subfolder="tokenizer"
        )

        print("Loading T5TokenizerFast")
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            model_name,
            subfolder="tokenizer_2"
        )

        # Initialize ControlNet
        print("Loading FluxControlNetModel")
        flux_controlnet = FluxControlNetModel.from_transformer(
            transformer,
            attention_head_dim=transformer.config["attention_head_dim"],
            num_attention_heads=transformer.config["num_attention_heads"],
            num_layers=num_double_layers,
            num_single_layers=num_single_layers,
        )
        flux_controlnet.enable_gradient_checkpointing()
        
        # Apply LoRA to ControlNet if requested
        if use_controlnet_lora:
            print("Applying LoRA to FluxControlNetModel")
            controlnet_target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",  # Attention layers
                "ff.net.0.proj", "ff.net.2",        # MLP layers
            ]
            controlnet_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=controlnet_target_modules,
                lora_dropout=0.0,
                bias="none",
            )
            flux_controlnet = get_peft_model(flux_controlnet, controlnet_lora_config)
            flux_controlnet.print_trainable_parameters()
        
        # Move models to GPU
        print("Moving models to GPU")
        transformer = transformer.to(device)
        vae = vae.to(device)
        text_encoder = text_encoder.to(device)
        text_encoder_2 = text_encoder_2.to(device)
        flux_controlnet = flux_controlnet.to(device)

        # Return all the models together
        return (transformer, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, flux_controlnet)
    return (load_models,)


@app.cell
def _():
    # Generic libs
    import os
    import math
    import random
    import gc
    import json
    from pathlib import Path
    from datetime import datetime
    from tqdm import tqdm

    # Data types and utils
    import torch

    # For F.mse_loss
    import torch.nn.functional as F

    # To load the datasets
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    import numpy as np
    import pandas as pd

    # Loading Models
    from diffusers import (
        FluxTransformer2DModel,
        FlowMatchEulerDiscreteScheduler,
        DDPMScheduler,
        AutoencoderKL,
        FluxPipeline
    )
    from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel
    from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
    from transformers import T5TokenizerFast, T5EncoderModel, CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig, get_peft_model, TaskType, get_peft_model_state_dict
    from einops import rearrange, repeat
    import bitsandbytes as bnb

    # Saving Data to Disk
    from safetensors.torch import save_file
    from PIL import Image

    # Saving Data to Oxen.ai (optional)
    from oxen import RemoteRepo
    return (
        AutoencoderKL,
        CLIPTextModel,
        CLIPTokenizer,
        DataLoader,
        Dataset,
        F,
        FlowMatchEulerDiscreteScheduler,
        FluxControlNetModel,
        FluxControlNetPipeline,
        FluxPipeline,
        FluxTransformer2DModel,
        Image,
        LoraConfig,
        RemoteRepo,
        T5EncoderModel,
        T5TokenizerFast,
        bnb,
        datetime,
        gc,
        get_peft_model,
        json,
        os,
        pd,
        random,
        rearrange,
        torch,
        transforms,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
