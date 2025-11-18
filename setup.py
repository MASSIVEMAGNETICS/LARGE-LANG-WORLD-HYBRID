"""Setup script for Large Language-World Hybrid AI."""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="large-lang-world-hybrid",
    version="1.0.0",
    author="MASSIVE MAGNETICS",
    description="A revolutionary Large Language-World Model Hybrid AI system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires='>=2.7',
    install_requires=[
        'numpy>=1.16.0,<1.17.0',
        'torch>=1.4.0,<1.8.0',
        'transformers>=2.11.0,<3.0.0',
        'Pillow>=6.2.0,<7.0.0',
    ],
    entry_points={
        'console_scripts': [
            'llwh-gui=llwh.gui.main:main',
            'llwh-train=llwh.training.trainer:main',
        ],
    },
)
