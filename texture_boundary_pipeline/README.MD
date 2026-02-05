# Texture Boundary Detection Pipeline

A modular, scalable pipeline for detecting texture boundaries in images using Vision-Language Models (VLMs).

---

## 🎯 Features

- ✅ **Model-agnostic** - Swap VLMs easily (Qwen, LLaVA, etc.)
- ✅ **Config-driven** - Prompts in Python for easy editing
- ✅ **Two pipelines** - Basic and Iterative (with classification)
- ✅ **IoU filtering** - Remove overlapping bounding boxes
- ✅ **Boundary fixing** - Smart edge detection near image boundaries
- ✅ **Size categorization** - Crops organized by size (tiny/small/medium/large/xlarge)
- ✅ **CLI interface** - Easy command-line usage
- ✅ **Modular design** - Reusable components

---

## 📁 Structure

```
texture_boundary_pipeline/
│
├── config/
│   ├── prompts.py              # All prompt definitions
│   └── PROMPTS_USAGE.md        # How to edit prompts
│
├── models/
│   ├── base_vlm.py             # Abstract VLM interface
│   ├── qwen_vlm.py             # Qwen implementation
│   ├── llava_vlm.py            # LLaVA placeholder
│   ├── model_factory.py        # Create models by name
│   └── __init__.py
│
├── utils/
│   ├── image_utils.py          # Image operations (padding, resizing)
│   ├── bbox_utils.py           # Bbox operations (IoU, parsing)
│   ├── io_utils.py             # File I/O, JSON operations
│   └── __init__.py
│
├── core/
│   ├── grounding.py            # Grounding logic
│   ├── bbox_processing.py      # IoU filtering + boundary fixing
│   ├── crop_extraction.py      # Extract crops by size
│   ├── visualization.py        # Draw bounding boxes
│   ├── classification.py       # Semantic vs texture classification
│   └── __init__.py
│
├── pipelines/
│   ├── basic_pipeline.py       # Ground → Process → Visualize → Extract
│   ├── iterative_pipeline.py   # + Classification + Refinement
│   └── __init__.py
│
├── main.py                      # CLI entry point
├── check_dependencies.py        # Dependency checker
└── README.md                    # This file
```

---

## 🚀 Quick Start

### **Installation**

```bash
# Install dependencies
pip install torch transformers qwen-vl-utils pillow numpy

# Clone or download the pipeline
cd texture_boundary_pipeline

# Check dependencies
python check_dependencies.py
```

### **Run Basic Pipeline**

```bash
python main.py basic --images /path/to/images --model qwen
```

### **Run Iterative Pipeline**

```bash
python main.py iterative --images /path/to/images --model qwen \
    --semantic-threshold 7.0 --max-iterations 2
```

---

## 📖 Usage

### **CLI Commands**

#### **Check Dependencies**
```bash
python main.py --check
```

#### **Basic Pipeline**
```bash
python main.py basic \
    --images /path/to/images \
    --model qwen \
    --output results \
    --iou-threshold 0.6 \
    --target-size 1024
```

**Options:**
- `--images`: Input image directory (required)
- `--model`: Model to use (default: qwen)
- `--output`: Output directory (default: results)
- `--iou-threshold`: IoU threshold for filtering (default: 0.6)
- `--no-fix-boundaries`: Disable boundary fixing
- `--target-size`: Target size for padding (default: 1024)
- `--quiet`: Suppress progress output

#### **Iterative Pipeline**
```bash
python main.py iterative \
    --images /path/to/images \
    --model qwen \
    --semantic-threshold 7.0 \
    --max-iterations 2 \
    --crop-categories medium large xlarge
```

**Additional Options:**
- `--semantic-threshold`: Confidence threshold for semantic (default: 7.0)
- `--max-iterations`: Max refinement iterations (default: 1)
- `--crop-categories`: Categories to classify (default: medium large xlarge)

---

### **Python API**

#### **Basic Pipeline**
```python
from models import create_model
from pipelines import basic_pipeline

# Create model
model = create_model('qwen', device='cuda')

# Run pipeline
results = basic_pipeline(
    model=model,
    image_dir='/path/to/images',
    output_dir='results',
    iou_threshold=0.6,
    fix_boundaries=True,
    verbose=True
)

print(f"Results saved to: {results['output_dir']}")
```

#### **Iterative Pipeline**
```python
from models.model_factory import create_model
from pipelines import iterative_pipeline

# Create model
model = create_model('qwen', device='cuda')

# Run pipeline
results = run_iterative_pipeline(
    model=model,
    image_dir='/path/to/images',
    output_dir='results',
    semantic_threshold=7.0,
    max_iterations=2,
    verbose=True
)
```

#### **Custom Pipeline**
```python
from models.model_factory import create_model
from pipelines.basic_pipeline import BasicPipeline


# Create model
model = create_model('qwen')

# List images
images = list_images('/path/to/images')

# Create pipeline
pipeline = BasicPipeline(
    model=model,
    output_dir='results',
    iou_threshold=0.6,
    fix_boundaries=True,
    verbose=True
)

# Run
results = pipeline.run(images)
```

---

## 🎨 Pipelines

### **Basic Pipeline**

1. **Grounding** - Extract texture boundary bounding boxes
2. **Processing** - IoU filtering + boundary fixing
3. **Visualization** - Draw annotated images
4. **Extraction** - Extract and categorize crops

**Output:**
```
results/run_TIMESTAMP/
├── grounding_results.json      # Raw grounding output
├── processed_bboxes.json       # After IoU + fixing
├── pipeline_summary.json       # Complete summary
├── visualizations/             # Annotated images
│   ├── annotated_image1.jpg
│   └── ...
└── crops/                      # Extracted crops by size
    ├── tiny/
    ├── small/
    ├── medium/
    ├── large/
    └── xlarge/
```

### **Iterative Pipeline**

Extends basic pipeline with:
1. **Classification** - Classify crops as semantic vs texture
2. **Refinement** - Re-ground semantic crops with focused prompt
3. **(Optional)** Multiple iterations

**Additional Output:**
```
results/run_TIMESTAMP/
├── ... (basic pipeline outputs)
├── iteration_1/
│   ├── classification_results.json    # Crop classifications
│   └── refined_bboxes.json           # Refined boundaries
└── iterative_pipeline_summary.json
```

---

## 🔧 Configuration

### **Prompts**

Edit `config/prompts.py` to customize prompts:

```python
GROUNDING_PROMPT = """Your custom prompt here..."""
SEMANTIC_CLASSIFICATION_PROMPT = """..."""
SCORING_PROMPT = """..."""
FOCUSED_GROUNDING_PROMPT = """..."""
```

See `config/PROMPTS_USAGE.md` for details.

### **Models**

Swap models easily:

```python
from models import create_model

# Use Qwen 8B
model = create_model('qwen-8b')

# Use Qwen 2B
model = create_model('qwen-2b')

# Use LLaVA (when implemented)
model = create_model('llava')
```

Add custom models:

```python
from models import BaseVLM, register_model

class MyVLM(BaseVLM):
    def load_model(self):
        # Your implementation
        pass
    
    def generate(self, image, prompt, max_tokens=512, **kwargs):
        # Your implementation
        pass
    
    def batch_generate(self, images, prompts, max_tokens=512, **kwargs):
        # Your implementation
        pass

# Register
register_model('my-model', MyVLM)

# Use
model = create_model('my-model')
```

---

## 📊 Output

### **Summary JSON**

```json
{
  "pipeline": "basic",
  "timestamp": "20241228_140530",
  "config": {
    "iou_threshold": 0.6,
    "fix_boundaries": true,
    "target_size": 1024
  },
  "input": {
    "num_images": 10,
    "image_names": ["image1.jpg", "image2.jpg", ...]
  },
  "grounding": {
    "total_boxes": 85
  },
  "processing": {
    "input_boxes": 85,
    "filtered": 12,
    "fixed": 8,
    "output_boxes": 73
  },
  "crops": {
    "total_crops": 73,
    "category_counts": {
      "tiny": 10,
      "small": 15,
      "medium": 20,
      "large": 18,
      "xlarge": 10
    }
  }
}
```

---

## 🧪 Testing

```bash
# Check all dependencies
python check_dependencies.py

---

## 🐛 Troubleshooting

### **"No module named 'torch'"**
```bash
pip install torch transformers qwen-vl-utils
```

### **"Qwen3VLForConditionalGeneration not found"**
Your transformers is too old. Update:
```bash
pip install --upgrade transformers
```

### **Import errors**
Run diagnostics:
```bash
python debug_imports.py
```

See `IMPORT_TROUBLESHOOTING.md` for detailed help.



## 🙏 Acknowledgments

Built with:
- Qwen3-VL by Alibaba
- PyTorch
- Transformers by Hugging Face
- PIL/Pillow

---

## 📧 Contact

aviadcohz@gmail.com