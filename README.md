# OS-Atlas FiftyOne Integration

A robust FiftyOne model integration for OS-Atlas vision-language models, designed specifically for GUI agents and UI understanding tasks.

## Features

- **Multiple Vision Tasks**: Detection, OCR, keypoint detection, classification, VQA, and agentic actions
- **Robust Parsing**: Handles inconsistent model output formats automatically
- **Flexible Prompting**: Support for custom prompts and system prompts
- **Production Ready**: Built-in error handling and graceful degradation

## Supported Operations

| Operation | Description | Output Format |
|-----------|-------------|---------------|
| `detect` | Object/UI element detection | `fo.Detections` |
| `ocr` | Grounded text detection and recognition | `fo.Detections` |
| `point` | Keypoint detection for UI elements | `fo.Keypoints` |
| `classify` | UI classification and categorization | `fo.Classifications` |
| `vqa` | Visual question answering | Raw text |
| `agentic` | GUI agent action planning | `fo.Keypoints` with metadata |



## Installation

### Register the Model Source

```python
import fiftyone.zoo as foz

# Register the OS-Atlas model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/os_atlas", 
    overwrite=True
)
```

### Load a Model

**Base Model (recommended for development):**
```python
model = foz.load_zoo_model(
    "OS-Copilot/OS-Atlas-Base-7B",
    install_requirements=True
)
```

## Quick Start

### Load a UI Dataset

```python
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

# Load sample UI dataset
dataset = fouh.load_from_hub(
    "Voxel51/GroundUI-18k",
    max_samples=100
)
```

### Basic Usage Examples

#### Visual Question Answering
```python
model.operation = "vqa"
model.prompt = "Describe this screenshot and what the user might be doing."
dataset.apply_model(model, label_field="vqa_results")
```

#### UI Element Detection
```python
model.operation = "detect"
model.prompt = "Find all buttons and interactive elements"
dataset.apply_model(model, label_field="ui_detections")
```

#### Grounded OCR
```python
model.operation = "ocr"
model.prompt = "Extract all text from UI elements like buttons, menus, and labels"
dataset.apply_model(model, label_field="ocr_results")
```

#### Keypoint Detection
```python
model.operation = "point"
model.prompt = "Find the search button"
dataset.apply_model(model, label_field="keypoints")
```

#### Classification
```python
model.operation = "classify"
model.prompt = "Classify this UI as: mobile app, web browser, desktop application, or other"
dataset.apply_model(model, label_field="ui_type")
```

#### Agentic Actions
```python
model.operation = "agentic"
model.prompt = "Click on the login button"
dataset.apply_model(model, label_field="agent_actions")
```

## Advanced Usage

### Using Dataset Fields as Prompts

Use existing fields in your dataset as dynamic prompts:

```python
# Use the 'instruction' field from your dataset
dataset.apply_model(
    model, 
    prompt_field="instruction",  # Field containing prompts
    label_field="results"
)
```

### Custom System Prompts

Override default system prompts for specialized behavior:

```python
# Clear existing system prompt
model.system_prompt = None

# Set custom system prompt
model.system_prompt = """
You are a specialized UI accessibility analyzer. 
Focus on identifying elements that may be difficult 
for users with visual impairments to interact with.
"""

model.operation = "detect"
dataset.apply_model(model, label_field="accessibility_analysis")
```

### Dynamic Classification

Generate classification prompts from dataset metadata:

```python
# Extract unique platforms from dataset
platforms = dataset.distinct("platform")

# Create dynamic classification prompt
model.operation = "classify"
model.prompt = f"Which platform is this from? Choose exactly one: {platforms}"
dataset.apply_model(model, label_field="platform_classification")
```

## Understanding Outputs

### Detection Results
- **Bounding boxes**: Normalized coordinates in FiftyOne format `[x, y, width, height]`
- **Labels**: Descriptive labels for detected UI elements
- **Confidence**: Automatic confidence scoring

### Keypoint Results
- **Points**: Normalized `[x, y]` coordinates
- **Labels**: Descriptive labels for interaction points
- **Metadata**: Additional context for agentic actions

### Agentic Action Metadata
Agentic operations include rich metadata:
- `action`: Type of action (click, type, scroll, etc.)
- `thought`: Model's reasoning process
- `sequence_idx`: Action order in multi-step plans
- Action-specific parameters (content for typing, direction for scrolling, etc.)

### Classification Results
- **Labels**: Predicted categories
- **Thought**: Model's reasoning (when available)

## Model Behavior & Robustness

This integration handles several model output inconsistencies automatically:

- **Format Variations**: Supports both direct arrays `[{...}]` and wrapped objects `{"key": [{...}]}`
- **Coordinate Formats**: Handles tuples, lists, strings, and malformed coordinate syntax
- **Truncated Output**: Recovers partial results from incomplete JSON
- **Mixed Dimensionality**: Converts 2D points to bounding boxes when needed

## Visualization

Launch the FiftyOne App to visualize results:

```python
import fiftyone as fo

# Launch interactive app
session = fo.launch_app(dataset)

# View results for first sample
sample = dataset.first()
print(f"VQA: {sample.vqa_results}")
print(f"Detections: {len(sample.ui_detections.detections)} objects found")
```

## Requirements

### Automatic Installation
The integration will automatically install required dependencies when loading with `install_requirements=True`:

- `huggingface-hub` - Model downloading and management
- `transformers>=4.30.0` - Transformer model support  
- `torch>=1.12.0` - PyTorch backend
- `torchvision` - Computer vision utilities
- `qwen-vl-utils` - Qwen vision-language utilities
- `accelerate` - Model acceleration and optimization

### Manual Installation
If you prefer manual dependency management:

```bash
pip install huggingface-hub transformers torch torchvision qwen-vl-utils accelerate
```

## License

This integration and the OS-Atlas models are released under the **Apache 2.0 License**, making them suitable for both commercial and non-commercial use. See the [original model repositories](https://huggingface.co/OS-Copilot) for complete license details.

## Contributing

Contributions are welcome! Please submit issues and pull requests to the main repository.

## Citation

If you use this integration in your research, please cite the original OS-Atlas paper and FiftyOne:

```bibtex
@article{wu2024atlas,
        title={OS-ATLAS: A Foundation Action Model for Generalist GUI Agents},
        author={Wu, Zhiyong and Wu, Zhenyu and Xu, Fangzhi and Wang, Yian and Sun, Qiushi and Jia, Chengyou and Cheng, Kanzhi and Ding, Zichen and Chen, Liheng and Liang, Paul Pu and others},
        journal={arXiv preprint arXiv:2410.23218},
        year={2024}
      }

```