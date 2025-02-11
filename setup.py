from setuptools import setup, find_packages

setup(
    name="BobVLM",
    version="0.1.0",
    author="selfDotOsman",
    description="A Vision Language Model pipeline for BobVLM",
    url="https://github.com/selfDotOsman/BobVLM",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "Pillow>=9.0.0",
        "huggingface-hub>=0.19.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
