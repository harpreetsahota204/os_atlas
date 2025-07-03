
import os
import logging
import json
from PIL import Image
from typing import Dict, Any, List, Union, Optional
import re
import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.utils import is_flash_attn_2_available

from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

DEFAULT_VQA_SYSTEM_PROMPT = "You are a helpful assistant. You provide clear and concise answerss to questions about images. Report answers in natural language text in English."

DEFAULT_DETECTION_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in detecting and localizating meaningful visual elements. 

You can detect and localize objects, components, people, places, things, and UI elements in images using 2D bound boxes.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "detections": [
        {
            "bbox": (x1,y1),(x2,y2),
            "label": "descriptive label for the bounding box"
        },
        {
            "bbox": (x1,y1),(x2,y2),
            "label": "descriptive label for the bounding box"
        }
    ]
}
```
"""


DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You are a GUI agent specializing in classification.

Unless specifically requested for single-class output, multiple relevant classifications can be provided.

# Output format

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "classifications": [
        {
            "thought": ...,
            "label": "descriptive class label", ## Short, descriptive, and precise. Refer to your thought about the appropriate label.
        },
        {
            "thought": ...,
            "label": "descriptive class label", ## Short, descriptive, and precise. Refer to your thought about the appropriate label.
        }
    ]
}
```
"""


DEFAULT_KEYPOINT_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in key point detection across any visual domain. A key point represents the center of any meaningful visual element. 

Key points should adapt to the context (physical world, digital interfaces, UI elements, etc.) while maintaining consistent accuracy and relevance. 

For each key point identify the key point and provide a contextually appropriate label and always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "keypoints": [
        {
            "point_2d": [x, y],
            "label": "descriptive label for the point"
        }
    ]
}
```

The JSON should contain points in pixel coordinates [x,y] format, where:
- x is the horizontal center coordinate of the visual element
- y is the vertical center coordinate of the visual element
- Include all relevant elements that match the user's request
- You can point to multiple visual elements

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's instructions and point.
"""


DEFAULT_OCR_SYSTEM_PROMPT = """You are a helpful assistant specializing in text detection and recognition (OCR) in screenshots. Your can read, detect, and locate text from any UI screenshot.

Text in a UI screenshot into the following categories, including but not limited to: title, abstract, heading, paragraph, button, link, label, icon, menu item, etc.

Always return your response as valid JSON wrapped in ```json blocks.

```json
{
    "text_detections": [
        {
            "bbox": (x1,y1),(x2,y2),
            "text_type": "decide on the appropriate category for this text",  
            "text": "Exact text content found in this region",
        }
    ]
}
```

- You are only allowed to read each text once
- 'text_type' is important to get right, it's the text region category based on the document, 
- The 'text' field should be a string containing the exact text content found in the region

The user might give you a single word instruction, a query, a list of objects, or more complex instructions. Adhere to the user's perform the OCR detections.
"""

DEFAULT_AGENTIC_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

# Action Space
click: point_2d=(x1,y1)
left_double: point_2d=(x1,y1)
right_single: point_2d=(x1,y1)
long_press: point_2d=(x1,y1)
drag: Return a list of two points, one for the start and one for the end. point_2d=[(x1,y1), (x1,y1)]
hotkey: key='ctrl c'
type: point_2d=(x1,y1), content='xxx'
finished: point_2d=(x1,y1), content='xxx'
scroll: point_2d=(x1,y1), direction='down/up/right/left'
wait: point_2d=(x1,y1)
open_app: point_2d=(x1,y1), app_name=''
press_home: set "point_2d" to the center of the screen point_2d=(x1,y1)
press_back: set "point_2d" to the center of the screen point_2d=(x1,y1)

# Output Format

Thought: think about the action space, additional parameters needed, and their values 

Always return your actions as valid JSON wrapped in ```json blocks, following this structure:

```json
{
    "keypoints": [
        {
            "thought": ..., #recall your thought about the action space, additional parameters needed, and their values 
            "action": ...,
            "point_2d": (x1,y1),
            "parameter_name": ... ## Refer to your thought about additional parameters.
        }
    ]
}
```

Note: Include only parameters relevant to your chosen action. Keep thoughts in English and summarize your plan with the target element in one sentence.
"""

MIN_PIXELS = 256*28*28
MAX_PIXELS = 1024*28*28


OPERATIONS = {
    "detect": DEFAULT_DETECTION_SYSTEM_PROMPT,
    "ocr": DEFAULT_OCR_SYSTEM_PROMPT,
    "point": DEFAULT_KEYPOINT_SYSTEM_PROMPT,
    "classify": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
    "vqa": DEFAULT_VQA_SYSTEM_PROMPT,
    "agentic": DEFAULT_AGENTIC_PROMPT
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class OSAtlasModel(SamplesMixin, Model):
    """A FiftyOne model for running OS-Atlas vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        model_kwargs = {
            "device_map":self.device,
            }
        
        # Only set specific torch_dtype for CUDA devices
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
        )
        logger.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            size={
            'shortest_edge': MIN_PIXELS,  # Minimum dimension
            'longest_edge': MAX_PIXELS    # Maximum dimension
            }
        )

        self.model.eval()

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else OPERATIONS[self.operation]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    def _parse_json(self, s: str) -> Dict:
        """
        Parse JSON from potentially truncated model output, auto-detecting structure.
        
        Tries standard parsing first. If truncated, finds the first array key 
        in the JSON and extracts complete objects from that array.
        
        Args:
            s: Raw string containing JSON (may be in markdown blocks)
            
        Returns:
            Complete JSON if valid, otherwise dict with the detected array key
            containing any complete objects found before truncation.
        """
        # Extract JSON content from markdown code blocks if present
        if "```json" in s:
            s = s.split("```json")[1].split("```")[0].strip()  # Get content between ```json and ``` markers
        
        # Try standard JSON parsing first - this handles complete, valid JSON
        try:
            return json.loads(s)  # Return immediately if parsing succeeds
        except:
            pass  # If parsing fails, continue to recovery methods
        
        # Recovery method 1: Find the first array key pattern like "key_name": [
        array_match = re.search(r'"([^"]+)":\s*\[', s)  # Regex to find array key
        if not array_match:
            return {"items": []}  # Return empty fallback structure if no array pattern found
        
        array_key = array_match.group(1)  # Extract the actual key name from regex match
        
        # Find the exact position where array content begins
        array_pattern = f'"{array_key}": ['  # Reconstruct the exact pattern to search for
        array_start = s.find(array_pattern) + len(array_pattern)  # Calculate start position of array content
        array_content = s[array_start:]  # Extract just the array content portion
        
        # Recovery method 2: Extract complete JSON objects from the array by tracking nested braces
        objects = []  # Will hold successfully parsed objects
        depth = 0  # Track nesting level of braces
        start = -1  # Position where current object starts
        
        for i, c in enumerate(array_content):
            if c == '{':
                if depth == 0:
                    start = i  # Mark start of a new object
                depth += 1  # Increase nesting level
            elif c == '}':
                depth -= 1  # Decrease nesting level
                if depth == 0 and start >= 0:
                    try:
                        # Extract and parse complete object
                        objects.append(json.loads(array_content[start:i+1]))  # Parse individual object
                    except:
                        pass  # Skip invalid objects
        
        # Return recovered objects with original key structure
        return {array_key: objects}

    def _to_detections(self, boxes: List[Dict], image_width: int, image_height: int) -> fo.Detections:
        """Convert bounding boxes to FiftyOne Detections.
        
        Takes detection results and converts them to FiftyOne's format, including:
        - Coordinate normalization
        - Label extraction
        
        Args:
            boxes: Detection results, either:
                - List of detection dictionaries
                - Dictionary containing 'data'
            image_width: Original image width in pixels
            image_height: Original image height in pixels
        
        Returns:
            fo.Detections: FiftyOne Detections object containing all converted detections
        """
        detections = []
        
        # Handle nested dictionary structures
        if isinstance(boxes, dict):
            # Try to get data field, fall back to original dict if not found
            boxes = boxes.get("detections", boxes)
            if isinstance(boxes, dict):
                # If still a dict, try to find first list value
                boxes = next((v for v in boxes.values() if isinstance(v, list)), boxes)
        
        # Ensure we're working with a list of boxes
        boxes = boxes if isinstance(boxes, list) else [boxes]
        
        # Process each bounding box
        for box in boxes:
            try:
                # Extract bbox coordinates, checking both possible keys
                bbox = box.get('bbox', box.get('bbox_2d', None))
                if not bbox:
                    continue
                    
                # Convert pixel coordinates to normalized [0,1] coordinates
                x1, y1, x2, y2 = map(float, bbox)
                x = x1 / image_width  # Left coordinate
                y = y1 / image_height  # Top coordinate
                w = (x2 - x1) / image_width  # Width
                h = (y2 - y1) / image_height  # Height
                
                # Create FiftyOne Detection object
                detection = fo.Detection(
                    label=str(box.get("label", "object")),
                    bounding_box=[x, y, w, h]
                )
                detections.append(detection)
                    
            except Exception as e:
                logger.debug(f"Error processing box {box}: {e}")
                continue
                    
        return fo.Detections(detections=detections)

    def _to_ocr_detections(self, boxes: List[Dict], image_width: int, image_height: int) -> fo.Detections:
        """Convert OCR results to FiftyOne Detections.
        
        Takes OCR detection results and converts them to FiftyOne's format, including:
        - Coordinate normalization
        - Text content preservation
        - Text type categorization
        
        Args:
            boxes: OCR detection results, either:
                - List of OCR dictionaries
                - Dictionary containing 'data'
            image_width: Original image width in pixels
            image_height: Original image height in pixels
        
        Returns:
            fo.Detections: FiftyOne Detections object containing all converted OCR detections
        """
        detections = []
        
        # Handle nested dictionary structures
        if isinstance(boxes, dict):
            # Try to get data field, fall back to original dict if not found
            boxes = boxes.get("text_detections", boxes)
            if isinstance(boxes, dict):
                # If still a dict, try to find first list value (usually "text_detections")
                boxes = next((v for v in boxes.values() if isinstance(v, list)), boxes)
        
        # Ensure boxes is a list, even for single box input
        boxes = boxes if isinstance(boxes, list) else [boxes]
        
        # Process each OCR box
        for box in boxes:
            try:
                # Extract bbox coordinates, checking both possible keys
                bbox = box.get('bbox', box.get('bbox_2d', None))
                if not bbox:
                    continue
                    
                # Extract text content and type
                text = box.get('text')
                text_type = box.get('text_type', 'text')  # Default to 'text' if not specified
                
                # Skip if no text content
                if not text:
                    continue
                    
                # Convert pixel coordinates to normalized [0,1] coordinates
                x1, y1, x2, y2 = map(float, bbox)
                x = x1 / image_width  # Left coordinate
                y = y1 / image_height  # Top coordinate
                w = (x2 - x1) / image_width  # Width
                h = (y2 - y1) / image_height  # Height
                
                # Create FiftyOne Detection object
                detection = fo.Detection(
                    label=str(text_type),
                    bounding_box=[x, y, w, h],
                    text=str(text)
                )
                detections.append(detection)
                    
            except Exception as e:
                logger.debug(f"Error processing OCR box {box}: {e}")
                continue
                    
        return fo.Detections(detections=detections)
    
    def _to_agentic_keypoints(self, actions: Dict, image_width: int, image_height: int) -> fo.Keypoints:
        """Convert agentic actions to FiftyOne Keypoints.
        
        Args:
            actions: Dictionary containing keypoints with point_2d, action type, and additional parameters
            image_width: Original image width in pixels
            image_height: Original image height in pixels
        """
        keypoints = []
        
        # Handle nested dictionary structures
        if isinstance(points, dict):
            points = points.get("keypoints", points)
            if isinstance(points, dict):
                points = next((v for v in points.values() if isinstance(v, list)), points)
        
        
        # Ensure actions is a list
        actions = actions if isinstance(actions, list) else [actions]
        
        for idx, kp in enumerate(actions):
            try:
                # Extract the point coordinates
                point_2d = kp.get("point_2d")
                if not point_2d:
                    continue
                    
                # Extract thought and action
                thought = kp.get("thought", "")
                action_type = kp.get("action", "")
                
                # Process coordinates based on action type
                if action_type == "drag" and isinstance(point_2d, list):
                    # For drag, point_2d is a list of two points [(x1,y1), (x2,y2)]
                    start_x, start_y = map(float, point_2d[0])
                    end_x, end_y = map(float, point_2d[1])
                    
                    # Normalize coordinates
                    point = [start_x / image_width, start_y / image_height]
                    
                    # Create metadata with end point and other fields
                    metadata = {
                        "sequence_idx": idx,
                        "action": action_type,
                        "thought": thought,
                        "end_point": [end_x / image_width, end_y / image_height]
                    }
                else:
                    # For all other actions, point_2d is a single point (x,y)
                    if isinstance(point_2d, list):
                        x, y = map(float, point_2d[0])
                    else:
                        x, y = map(float, point_2d)
                    
                    # Normalize coordinates
                    point = [x / image_width, y / image_height]
                    
                    # Base metadata with sequence index, action type and thought
                    metadata = {
                        "sequence_idx": idx,
                        "action": action_type,
                        "thought": thought
                    }
                    
                    # Add action-specific parameters
                    if action_type == "type":
                        metadata["content"] = kp.get("content", "")
                        
                    elif action_type == "finished":
                        metadata["content"] = kp.get("content", "")
                        
                    elif action_type == "scroll":
                        metadata["direction"] = kp.get("direction", "down")
                        
                    elif action_type == "hotkey":
                        metadata["key"] = kp.get("key", "")
                        
                    elif action_type == "open_app":
                        metadata["app_name"] = kp.get("app_name", "")
                
                # Create FiftyOne Keypoint object
                keypoint = fo.Keypoint(
                    label=action_type,
                    points=[point],
                    metadata=metadata
                )
                keypoints.append(keypoint)
                    
            except Exception as e:
                logger.debug(f"Error processing keypoint {kp}: {e}")
                continue
                    
        return fo.Keypoints(keypoints=keypoints)

    def _to_keypoints(self, points: List[Dict], image_width: int, image_height: int) -> fo.Keypoints:
        """Convert keypoint detections to FiftyOne Keypoints.
        
        Processes keypoint coordinates and normalizes them to [0,1] range while
        preserving associated labels.
        
        Args:
            points: Keypoint detection results, either:
                - List of keypoint dictionaries
                - Dictionary containing 'data'
            image_width: Original image width in pixels
            image_height: Original image height in pixels
            
        Returns:
            fo.Keypoints: FiftyOne Keypoints object containing all converted keypoints
        """
        keypoints = []
        
        # Handle nested dictionary structures
        if isinstance(points, dict):
            points = points.get("keypoints", points)
            if isinstance(points, dict):
                points = next((v for v in points.values() if isinstance(v, list)), points)
        
        # Process each keypoint
        for point in points:
            try:
                # Extract and normalize coordinates
                x, y = point["point_2d"]
                # Handle tensor inputs if present
                x = float(x.cpu() if torch.is_tensor(x) else x)
                y = float(y.cpu() if torch.is_tensor(y) else y)
                
                # Normalize coordinates to [0,1] range
                normalized_point = [
                    x / image_width,
                    y / image_height
                ]
                
                # Create FiftyOne Keypoint object
                keypoint = fo.Keypoint(
                    label=str(point.get("label", "point")),
                    points=[normalized_point]
                )
                keypoints.append(keypoint)
            except Exception as e:
                logger.debug(f"Error processing point {point}: {e}")
                continue
                
        return fo.Keypoints(keypoints=keypoints)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert classification results to FiftyOne Classifications.
        
        Processes classification labels into FiftyOne's format.
        
        Args:
            classes: Classification results, either:
                - List of classification dictionaries
                - Dictionary containing 'data'
                
        Returns:
            fo.Classifications: FiftyOne Classifications object containing all results
        """
        classifications = []
        
        # Handle nested dictionary structures
        if isinstance(classes, dict):
            classes = classes.get("data", classes)
            if isinstance(classes, dict):
                classes = next((v for v in classes.values() if isinstance(v, list)), classes)
        
        # Process each classification
        for cls in classes:
            try:
                # Create FiftyOne Classification object
                classification = fo.Classification(
                    label=str(cls["label"])
                )
                classifications.append(classification)
            except Exception as e:
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        return fo.Classifications(classifications=classifications)

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Keypoints, fo.Classifications, str]:
        """Process a single image through the model and return predictions.
        
        This internal method handles the core prediction logic including:
        - Constructing the chat messages with system prompt and user query
        - Processing the image and text through the model
        - Parsing the output based on the operation type (detection/points/classification/VQA)
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            One of:
            - fo.Detections: For object detection results
            - fo.Keypoints: For keypoint detection results  
            - fo.Classifications: For classification results
            - str: For VQA text responses
            
        Raises:
            ValueError: If no prompt has been set
        """
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt  # Start with instance default
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)  # Local variable, doesn't affect instance
        
        if not prompt:
            raise ValueError("No prompt provided.")
        
        messages = [
            {
                "role": "system", 
                "content": [  
                    {
                        "type": "text",
                        "text": self.system_prompt
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image", 
                        "image": sample.filepath if sample else image
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ]
            }
        ]
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
            )
        
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text], 
            images=image_inputs,
            videos=video_inputs,
            padding=True, 
            return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=8192)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # Get image dimensions and convert to float
        input_height = float(inputs['image_grid_thw'][0][1].cpu() * 14)
        input_width = float(inputs['image_grid_thw'][0][2].cpu() * 14)

        # For VQA, return the raw text output
        if self.operation == "vqa":
            return output_text.strip()
        elif self.operation == "detect":
            parsed_output = self._parse_json(output_text)
            return self._to_detections(parsed_output, input_width, input_height)
        elif self.operation == "ocr":
            parsed_output = self._parse_json(output_text)
            return self._to_detections(parsed_output, input_width, input_height)
        elif self.operation == "point":
            parsed_output = self._parse_json(output_text)
            return self._to_keypoints(parsed_output, input_width, input_height)
        elif self.operation == "classify":
            parsed_output = self._parse_json(output_text)
            return self._to_classifications(parsed_output)
        elif self.operation == "agentic":  # Fixed missing colon
            parsed_output = self._parse_json(output_text)
            return self._to_agentic_keypoints(parsed_output, input_width, input_height)  # Should use _to_agentic_keypoints

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)