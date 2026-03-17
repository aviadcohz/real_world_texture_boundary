from setuptools import setup, find_packages

setup(
    name="texture_boundary_Architexture",
    version="1.0.0",
    description="Automated texture transition boundary extraction from segmented images",
    author="Aviad",
    packages=find_packages(where=".."),
    package_dir={"": ".."},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "Pillow>=10.0",
        "opencv-python>=4.8",
        "scikit-image>=0.21",
        "torch>=2.1",
        "transformers>=4.40",
        "qwen-vl-utils>=0.0.8",
        "accelerate>=0.30",
    ],
    extras_require={
        "sr": ["realesrgan>=0.3.0", "basicsr>=1.4.2"],
    },
)
