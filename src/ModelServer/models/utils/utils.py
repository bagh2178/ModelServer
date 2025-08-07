import os
import json
import cv2
from moviepy.editor import ImageSequenceClip
import numpy as np
from PIL import Image


class Utils:
    def __init__(self):
        pass

    def save_image(self, image, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pass
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        image.save(path)

    def save_text(self, text, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            f.write(text)

    def save_json(self, json_data, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            json.dump(json_data, f, indent=4)
    
    def save_video(self, image_list, video_path, fps=30, resize_factor=1.0, input_color_space="RGB"):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        first_frame = image_list[0]
        original_height, original_width = first_frame.shape[:2]

        new_width = int(original_width * resize_factor)
        new_height = int(original_height * resize_factor)

        new_width = new_width if new_width % 2 == 0 else new_width - 1
        new_height = new_height if new_height % 2 == 0 else new_height - 1

        new_width = max(2, new_width)
        new_height = max(2, new_height)

        resized_images = []
        for img in image_list:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            if img.shape[-1] == 3:
                if input_color_space == "BGR":
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif input_color_space == "RGB":
                    pass
                else:
                    raise ValueError("Unsupported input_color_space. Use 'RGB' or 'BGR'.")

            resized_img = cv2.resize(
                img,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
            resized_images.append(resized_img)

        clip = ImageSequenceClip(resized_images, fps=fps)
        clip.write_videofile(
            video_path,
            codec="libx264",
            audio=False,
            verbose=False,
            logger=None,
            threads=4,
            ffmpeg_params=["-pix_fmt", "yuv420p"]
        )

    def stack_images(self, image_list, scale_list=None, text=None):
        """
        Stack multiple images vertically with individual scaling while maintaining aspect ratios.
        
        Args:
            image_list: List of images to stack (will be converted to numpy arrays)
            scale_list: List of scale factors for each image (will be normalized so max value becomes 1.0).
                    If None, defaults to all 1.0
            text: Optional text to display above all images
        
        Returns:
            Stacked image with white padding where needed (numpy array)
        """
        if len(image_list) == 0:
            raise ValueError("image_list cannot be empty")
        
        # Default scale_list to all 1.0 if not provided
        if scale_list is None:
            scale_list = [1.0] * len(image_list)
        
        if len(image_list) != len(scale_list):
            raise ValueError("image_list and scale_list must have the same length")
        
        # Convert all images to numpy arrays first
        numpy_images = []
        for img in image_list:
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            numpy_images.append(img)
        
        # Process images: ensure all have 3 channels
        processed_images = []
        for img in numpy_images:
            # Handle grayscale images
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            processed_images.append(img)
        
        # Normalize scale_list so that the maximum value is 1.0
        max_scale_value = max(scale_list)
        normalized_scales = [scale / max_scale_value for scale in scale_list]
        
        # Find the maximum scale (which is now 1.0) to determine the base width
        max_scale = max(normalized_scales)
        
        # Calculate the base width from the image with maximum scale
        max_scale_indices = [i for i, scale in enumerate(normalized_scales) if scale == max_scale]
        # Use the first image with max scale to determine base width
        base_img = processed_images[max_scale_indices[0]]
        base_width = base_img.shape[1]
        
        # Process each image according to its normalized scale
        final_images = []
        for img, scale in zip(processed_images, normalized_scales):
            h, w = img.shape[:2]
            aspect_ratio = h / w
            
            # Calculate the target width for this image based on its scale
            target_width = int(base_width * scale)
            target_height = int(target_width * aspect_ratio)
            
            # Resize the image maintaining aspect ratio
            resized_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            # Create a white background with the base width
            padded_img = np.ones((target_height, base_width, 3), dtype=resized_img.dtype) * 255
            
            # Center the resized image horizontally
            start_x = (base_width - target_width) // 2
            end_x = start_x + target_width
            padded_img[:, start_x:end_x] = resized_img
            
            final_images.append(padded_img)
        
        # Add text area if text is provided
        if text is not None:
            # Create text area
            text_height = 80  # Height for text area
            text_area = np.ones((text_height, base_width, 3), dtype=final_images[0].dtype) * 255
            
            # Add text to the text area
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = (0, 0, 0)  # Black text
            
            # Get text size to center it
            (text_width, text_baseline), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            
            # Calculate position to center text
            x = (base_width - text_width) // 2
            y = (text_height + text_baseline) // 2
            
            # Put text on the text area
            cv2.putText(text_area, text, (x, y), font, font_scale, text_color, font_thickness)
            
            # Insert text area at the beginning
            final_images.insert(0, text_area)
        
        # Stack all processed images vertically (including text area if present)
        stacked_image = np.concatenate(final_images, axis=0)
        
        return stacked_image