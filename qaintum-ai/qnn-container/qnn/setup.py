from setuptools import setup, find_packages

setup(
    name="qnn",
    version="0.1.0",
    author="qAIntum.ai",
    description="A Quantum Neural Network package for classification and regression tasks.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/qaintum-ai/qnn-container",
    packages=find_packages(),
    install_requires=[
        "torch==2.2.2",
        "torchvision==0.17.2",
        "PennyLane==0.29.1",
        "PennyLane-SF==0.29.1",
        "numpy==1.23.5",
        "scipy==1.10.1",
        "setuptools==65.5.0",
        "tqdm>=4.60",
        "matplotlib>=3.4",
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
