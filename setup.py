"""
PIVOT Setup Script
"""

from setuptools import setup, find_packages

setup(
    name="pivot-navigation",
    version="0.1.0",
    description="PIVOT: Prompting with Iterative Visual Optimization for Drone Navigation",
    author="PIVOT Team",
    python_requires=">=3.10,<3.12",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "airsim>=1.8.1",
        "backports-ssl-match-hostname>=3.7.0.1",
        "djitellopy>=2.5.0",
        "google-genai>=1.35.0",
        "matplotlib>=3.10.6",
        "msgpack<1.0",  # AirSim 1.8.1 expects msgpack encoding kwargs
        "mss>=10.1.0",
        "numpy>=1.26,<2.3",
        "openai>=1.107.0",
        "opencv-python>=4.11.0.86",
        "pillow>=11.3.0",
        "pynput>=1.8.1",
        "python-dotenv>=1.1.1",
        "pyyaml>=6.0.2",
    ],
    entry_points={
        "console_scripts": [
            "pivot=pivot.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
