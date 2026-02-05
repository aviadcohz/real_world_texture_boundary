"""
Example: How to use the models package

This file demonstrates how to use the VLM models.
"""

# Example 1: Create model using factory
from models.model_factory import create_model

# Create Qwen model (easiest way)
model = create_model('qwen', device='cuda')

# Alternative: Create specific Qwen variant
model_8b = create_model('qwen-8b', device='cuda')
model_2b = create_model('qwen-2b', device='cuda')


# Example 2: Use model directly
from models.qwen_vlm import QwenVLM

model = QwenVLM(
    model_name="Qwen/Qwen3-VL-8B-Instruct",
    device="cuda"
)


# Example 3: Generate response
image_path = "/path/to/image.jpg"
prompt = "Describe the textures in this image."

response = model.generate(image_path, prompt, max_tokens=512)
print(response)


# Example 4: Batch generation
images = [
    "/path/to/image1.jpg",
    "/path/to/image2.jpg",
    "/path/to/image3.jpg"
]

prompts = [
    "Find texture boundaries",
    "Find texture boundaries", 
    "Find texture boundaries"
]

responses = model.batch_generate(images, prompts)
for img, resp in zip(images, responses):
    print(f"{img}: {resp[:100]}...")


# Example 5: List available models
from models.model_factory import list_available_models, get_model_info

print("\nAvailable models:")
for model_name in list_available_models():
    info = get_model_info(model_name)
    print(f"  {model_name}: {info}")


# Example 6: Register custom model
from models.base_vlm import BaseVLM
from models.model_factory import register_model

class MyCustomVLM(BaseVLM):
    def load_model(self):
        print("Loading my custom model...")
        self._model = "dummy_model"
    
    def generate(self, image, prompt, max_tokens=512, **kwargs):
        return "Custom response"
    
    def batch_generate(self, images, prompts, max_tokens=512, **kwargs):
        return ["Custom response"] * len(images)

# Register it
register_model('my-custom', MyCustomVLM)

# Now you can use it
custom_model = create_model('my-custom')


# Example 7: Model info
model = create_model('qwen')
print(f"\nModel info: {model.info}")
print(f"Is loaded: {model.is_loaded}")
print(f"Representation: {model}")