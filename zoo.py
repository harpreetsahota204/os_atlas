
import os
import logging
import json
from PIL import Image
from typing import List, Tuple, Dict, Any, List, Union, Optional
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
            "bbox": (x1,y1,x2,y2),
            "label": "descriptive label for the bounding box"
        },
        {
            "bbox": (x1,y1,x2,y2),
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
            "point_2d": (x1,y1),
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
            "bbox": (x1,y1,x2,y2),
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
        """Parse JSON from potentially truncated model output, auto-detecting structure.
        
        Tries standard parsing first. If truncated, finds the first array key 
        and extracts complete objects from that array.
        
        Args:
            s: Raw string containing JSON (may be in markdown blocks)
            
        Returns:
            Complete JSON if valid, otherwise dict with the detected array key
            containing any complete objects found before truncation.
        """
        # Extract JSON from markdown blocks
        if "```json" in s:
            s = s.split("```json")[1].split("```")[0].strip()
        
        # Try standard parsing first
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # JSON parsing failed - try to fix coordinate formatting issues first
            try:
                # Fix the specific issue: (x,y),(x2,y2) -> (x,y), (x2,y2)
                # Add space after ),( pattern
                fixed_s = re.sub(r'\),\(', '), (', s)
                return json.loads(fixed_s)
            except json.JSONDecodeError:
                pass
        
        # If JSON parsing still fails, fall back to object extraction
        # This handles both malformed JSON and truncated output
        array_match = re.search(r'"([^"]+)":\s*\[', s)
        if not array_match:
            return {"items": []}
        
        array_key = array_match.group(1)
        
        # Extract array content
        array_start = s.find(f'"{array_key}": [') + len(f'"{array_key}": [')
        array_content = s[array_start:]
        
        # Extract complete objects by tracking brace depth
        objects = []
        depth = 0
        obj_start = None
        
        for i, char in enumerate(array_content):
            if char == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        obj_json = array_content[obj_start:i+1]
                        # Apply coordinate fixes to individual objects too
                        obj_json = re.sub(r'\),\(', '), (', obj_json)
                        objects.append(json.loads(obj_json))
                    except json.JSONDecodeError:
                        # If individual object parsing fails, skip it
                        # The coordinate parsing methods will handle malformed coordinates
                        pass
                    obj_start = None
        
        return {array_key: objects}

    def _extract_list_from_data(self, data: Union[Dict, List, Any], key: str) -> List[Any]:
        """Extract list from various nested data structures.
        
        Handles both direct array formats and wrapped formats:
        - Direct: [ { "item": "data" } ]
        - Wrapped: { "key": [ { "item": "data" } ] }
        """
        # If it's already a list, return it directly (handles direct array format)
        if isinstance(data, list):
            return data
            
        # If it's a dict, try to extract the list from various nested structures
        if isinstance(data, dict):
            data = data.get(key, data)
            if isinstance(data, dict):
                data = next((v for v in data.values() if isinstance(v, list)), data)
        
        # Ensure we always return a list
        return data if isinstance(data, list) else [data]

    def _parse_bbox_coords(self, bbox: Union[List, Tuple, str, Any]) -> Optional[Tuple[float, float, float, float]]:
        """Parse bounding box coordinates from various formats.
        
        Args:
            bbox: Bounding box coordinates in various possible formats:
                - List/tuple of 4 numbers [x1, y1, x2, y2]
                - String representation like "(x1,y1),(x2,y2)" or "x1 y1 x2 y2"
                - Other formats that can be converted to 4 coordinates
        
        Returns:
            Tuple of (x1, y1, x2, y2) as floats, or None if parsing fails
        """
        try:
            # Case 1: If it's already a list/tuple with 4 elements, use it directly
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                return tuple(map(float, bbox))
            
            # Case 2: For strings or other formats, extract all numbers and take the first 4
            numbers = re.findall(r'-?\d+(?:\.\d+)?', str(bbox))
            if len(numbers) >= 4:
                return tuple(map(float, numbers[:4]))
                
        except (ValueError, TypeError) as e:
            logger.debug(f"Error processing bbox {bbox}: {e}")
            
        return None

    def _parse_point_coords(self, point: Union[List, Tuple, str, Any]) -> Optional[Tuple[float, float]]:
        """Parse point coordinates from various formats.
        
        Args:
            point: Point coordinates in various possible formats:
                - List/tuple of 2 numbers [x, y]
                - String representation like "(x,y)" or "x y"
                - Other formats that can be converted to 2 coordinates
        
        Returns:
            Tuple of (x, y) as floats, or None if parsing fails
        """
        try:
            # Case 1: If it's already a list/tuple with 2 elements, use it directly
            if isinstance(point, (list, tuple)) and len(point) == 2:
                return tuple(map(float, point))
            
            # Case 2: For strings or other formats, extract all numbers and take the first 2
            numbers = re.findall(r'-?\d+(?:\.\d+)?', str(point))
            if len(numbers) >= 2:
                return tuple(map(float, numbers[:2]))
                
        except (ValueError, TypeError) as e:
            logger.debug(f"Error processing point {point}: {e}")
            
        return None


    def _convert_bbox_to_fiftyone(self, bbox: Any) -> Optional[List[float]]:
        """Convert bbox coordinates to FiftyOne format.
        
        Converts from model's 0-1000 normalized coordinates to FiftyOne's 0-1 format.
        Model outputs: [x1, y1, x2, y2] in 0-1000 range (top-left, bottom-right)
        FiftyOne expects: [top-left-x, top-left-y, width, height] in 0-1 range
        
        Args:
            bbox: Bounding box coordinates in 0-1000 range
            
        Returns:
            List of [x, y, width, height] in 0-1 range for FiftyOne, or None if invalid
        """
        coords = self._parse_bbox_coords(bbox)
        if not coords or len(coords) != 4:
            return None
            
        x1_norm, y1_norm, x2_norm, y2_norm = coords
        
        # Convert from 0-1000 range to 0-1 range for FiftyOne
        x = x1_norm / 1000.0  # Left coordinate (0-1)
        y = y1_norm / 1000.0  # Top coordinate (0-1)
        w = (x2_norm - x1_norm) / 1000.0  # Width (0-1)
        h = (y2_norm - y1_norm) / 1000.0  # Height (0-1)
        
        return [x, y, w, h]

    def _convert_point_to_fiftyone(self, point: Any) -> Optional[List[float]]:
        """Convert point coordinates to FiftyOne format.
        
        Converts from model's 0-1000 normalized coordinates to FiftyOne's 0-1 format.
        
        Args:
            point: Point coordinates in 0-1000 range
            
        Returns:
            List of [x, y] in 0-1 range for FiftyOne, or None if invalid
        """
        coords = self._parse_point_coords(point)
        if not coords or len(coords) != 2:
            return None
            
        x_norm, y_norm = coords
        
        # Convert from 0-1000 range to 0-1 range for FiftyOne
        x = x_norm / 1000.0  # X coordinate (0-1)
        y = y_norm / 1000.0  # Y coordinate (0-1)
        
        return [x, y]

    def _to_detections(self, boxes: List[Dict]) -> fo.Detections:
        """Convert bounding boxes to FiftyOne Detections.
        
        Takes the raw model output containing bounding box information and converts
        it to FiftyOne's Detection objects that can be visualized in the UI.
        
        Args:
            boxes: List of dictionaries containing detection information with 'bbox' and 'label' keys
            
        Returns:
            fo.Detections: A FiftyOne Detections object containing all valid detections
        """
        detections = []
        # Extract the 'detections' list from the model output if nested in a container
        boxes = self._extract_list_from_data(boxes, "detections")
        
        for box in boxes:
            try:
                # Get bbox coordinates - try both 'bbox' and 'bbox_2d' keys for flexibility
                bbox = box.get('bbox', box.get('bbox_2d'))
                if not bbox:
                    # Skip entries without bounding box information
                    continue
                    
                # Convert from model's coordinate system to FiftyOne's expected format
                fiftyone_bbox = self._convert_bbox_to_fiftyone(bbox)
                if not fiftyone_bbox:
                    # Log and skip if conversion failed (invalid coordinates)
                    logger.debug(f"Invalid bbox format: {bbox}")
                    continue
                
                # Create a FiftyOne Detection object with the converted coordinates
                # Default to "object" label if none provided
                detection = fo.Detection(
                    label=str(box.get("label", "object")),
                    bounding_box=fiftyone_bbox
                )
                detections.append(detection)
                    
            except Exception as e:
                # Catch and log any errors during processing of individual boxes
                logger.debug(f"Error processing box {box}: {e}")
                continue
                    
        # Return a FiftyOne Detections object containing all valid detections
        return fo.Detections(detections=detections)
    
    def _to_ocr_detections(self, boxes: List[Dict]) -> fo.Detections:
        """Convert OCR results to FiftyOne Detections.
        
        Takes the raw model output containing OCR text detection information and converts
        it to FiftyOne's Detection objects that can be visualized in the UI.
        
        Args:
            boxes: List of dictionaries containing OCR detection information with 'bbox', 
                  'text', and optionally 'text_type' keys
            
        Returns:
            fo.Detections: A FiftyOne Detections object containing all valid OCR text detections
        """
        detections = []
        # Extract the 'text_detections' list from the model output if nested in a container
        boxes = self._extract_list_from_data(boxes, "text_detections")
        
        for box in boxes:
            try:
                # Get bbox and text content - try both possible key names for flexibility
                bbox = box.get('bbox', box.get('bbox_2d'))
                text = box.get('text')
                # Skip entries without both bounding box and text content
                if not bbox or not text:
                    continue
                    
                # Convert from model's coordinate system to FiftyOne's expected format
                fiftyone_bbox = self._convert_bbox_to_fiftyone(bbox)
                if not fiftyone_bbox:
                    # Log and skip if conversion failed (invalid coordinates)
                    logger.debug(f"Invalid bbox format: {bbox}")
                    continue
                
                # Create a FiftyOne Detection object with the converted coordinates
                # Use 'text_type' as label if provided, otherwise default to "text"
                detection = fo.Detection(
                    label=str(box.get('text_type', 'text')),
                    bounding_box=fiftyone_bbox,
                    text=str(text)  # Store the actual OCR text content
                )
                detections.append(detection)
                    
            except Exception as e:
                # Catch and log any errors during processing of individual OCR boxes
                logger.debug(f"Error processing OCR box {box}: {e}")
                continue
                    
        # Return a FiftyOne Detections object containing all valid OCR text detections
        return fo.Detections(detections=detections)

    def _to_keypoints(self, points: List[Dict]) -> fo.Keypoints:
        """Convert keypoint detections to FiftyOne Keypoints.
        
        Takes the raw model output containing keypoint detection information and converts
        it to FiftyOne's Keypoint objects that can be visualized in the UI.
        
        Args:
            points: List of dictionaries containing keypoint information with 'point_2d' 
                   and optionally 'label' keys
            
        Returns:
            fo.Keypoints: A FiftyOne Keypoints object containing all valid keypoint detections
        """
        keypoints = []
        # Extract the 'keypoints' list from the model output if nested in a container
        points = self._extract_list_from_data(points, "keypoints")
        
        for point in points:
            try:
                # Convert from model's coordinate system to FiftyOne's expected format
                fiftyone_point = self._convert_point_to_fiftyone(point["point_2d"])
                if not fiftyone_point:
                    # Log and skip if conversion failed (invalid coordinates)
                    logger.debug(f"Invalid point coordinates: {point['point_2d']}")
                    continue
                
                # Create a FiftyOne Keypoint object with the converted coordinates
                # Use provided label if available, otherwise default to "point"
                keypoint = fo.Keypoint(
                    label=str(point.get("label", "point")),
                    points=[fiftyone_point]  # FiftyOne expects points as a list
                )
                keypoints.append(keypoint)
                
            except Exception as e:
                # Catch and log any errors during processing of individual keypoints
                logger.debug(f"Error processing point {point}: {e}")
                continue
                
        # Return a FiftyOne Keypoints object containing all valid keypoint detections
        return fo.Keypoints(keypoints=keypoints)

    def _to_agentic_keypoints(self, actions: Dict) -> fo.Keypoints:
        """Convert agentic actions to FiftyOne Keypoints.
        
        Processes the model's agentic action outputs and converts them into FiftyOne Keypoint
        objects that can be visualized in the UI. Handles different action types including
        clicks, drags, typing, scrolling, and other UI interactions.
        
        Args:
            actions: Dictionary containing agentic actions, typically with a "keypoints" key
                    holding a list of action dictionaries
            
        Returns:
            fo.Keypoints: A FiftyOne Keypoints object containing all valid action keypoints
                         with appropriate metadata for visualization and interaction
        """
        keypoints = []
        # Extract the list of keypoint actions from potentially nested structure
        actions = self._extract_list_from_data(actions, "keypoints")
        
        for idx, kp in enumerate(actions):
            try:
                # Get the point coordinates for this action
                point_2d = kp.get("point_2d")
                if not point_2d:
                    # Skip actions without valid coordinates
                    continue
                    
                action_type = kp.get("action", "")
                
                if action_type == "drag" and isinstance(point_2d, list):
                    # Special handling for drag actions which have start and end points
                    # Convert both points from model coordinates to FiftyOne format
                    start_point = self._convert_point_to_fiftyone(point_2d[0])
                    end_point = self._convert_point_to_fiftyone(point_2d[1])
                    if not start_point or not end_point:
                        # Skip if either point conversion failed
                        logger.debug(f"Invalid drag coordinates: {point_2d}")
                        continue
                        
                    # Store drag-specific metadata including the end point
                    metadata = {
                        "sequence_idx": idx,  # Track action sequence order
                        "action": action_type,
                        "thought": kp.get("thought", ""),  # Agent's reasoning
                        "end_point": end_point  # Store end point in metadata
                    }
                    point = start_point  # Use start point as the main keypoint location
                else:
                    # Handle all other single-point actions (click, type, etc.)
                    # Handle both list and tuple coordinate formats
                    coords = point_2d[0] if isinstance(point_2d, list) else point_2d
                    point = self._convert_point_to_fiftyone(coords)
                    if not point:
                        # Skip if point conversion failed
                        logger.debug(f"Invalid point coordinates: {coords}")
                        continue
                        
                    # Build base metadata common to all action types
                    metadata = {
                        "sequence_idx": idx,  # Track action sequence order
                        "action": action_type,
                        "thought": kp.get("thought", "")  # Agent's reasoning
                    }
                    
                    # Add action-specific parameters to metadata
                    if action_type == "type":
                        # For typing actions, include the text content
                        metadata["content"] = kp.get("content", "")
                    elif action_type == "finished":
                        # For task completion, include any final message
                        metadata["content"] = kp.get("content", "")
                    elif action_type == "scroll":
                        # For scrolling, include direction (up/down/left/right)
                        metadata["direction"] = kp.get("direction", "down")
                    elif action_type == "hotkey":
                        # For keyboard shortcuts, include the key combination
                        metadata["key"] = kp.get("key", "")
                    elif action_type == "open_app":
                        # For app launching, include the app name
                        metadata["app_name"] = kp.get("app_name", "")
                
                # Create the FiftyOne Keypoint object with all metadata
                keypoint = fo.Keypoint(
                    label=action_type,  # Use action type as the label
                    points=[point],  # FiftyOne expects points as a list
                    metadata=metadata  # Include all action metadata
                )
                keypoints.append(keypoint)
                    
            except Exception as e:
                # Catch and log any errors during processing of individual actions
                logger.debug(f"Error processing keypoint {kp}: {e}")
                continue
                    
        # Return a FiftyOne Keypoints object containing all valid action keypoints
        return fo.Keypoints(keypoints=keypoints)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert classification results to FiftyOne Classifications."""
        classifications = []
        classes = self._extract_list_from_data(classes, "classifications")
        
        for cls in classes:
            try:
                label = cls.get("label")
                thought = cls.get("thought", "")

                classification = fo.Classification(
                    label=label,
                    thought=thought
                )
                classifications.append(classification)
                
            except Exception as e:
                logger.debug(f"Error processing classification {cls}: {e}")
                continue
                
        return fo.Classifications(classifications=classifications)

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Keypoints, fo.Classifications, str]:
        """Process a single image through the model and return predictions."""
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
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=8192)
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=True
        )[0]

        # For VQA, return the raw text output
        if self.operation == "vqa":
            return output_text.strip()
        
        # For all other operations, parse JSON and convert to appropriate format
        print(f"====RAW MODEL OUTPUT: {output_text}====")
        parsed_output = self._parse_json(output_text)
        
        if self.operation == "detect":
            return self._to_detections(parsed_output)
        elif self.operation == "ocr":
            return self._to_ocr_detections(parsed_output)
        elif self.operation == "point":
            return self._to_keypoints(parsed_output)
        elif self.operation == "classify":
            return self._to_classifications(parsed_output)
        elif self.operation == "agentic": 
            return self._to_agentic_keypoints(parsed_output)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

    def predict(self, image, sample=None):
        """Process an image with the model."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)