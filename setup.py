from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="medvision-ai-agent",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive AI agent for medical image analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/medvision-ai-agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "medvision-train=main:train_main",
            "medvision-predict=main:predict_main",
            "medvision-serve=main:serve_main",
        ],
    },
    include_package_data=True,
    package_data={
        "medvision": ["config/*.yaml", "models/*.h5"],
    },
)
