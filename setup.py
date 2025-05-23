from setuptools import setup, find_packages

setup(
    name="inference",
    version="0.1.0",
    description="CLI for Emotion Classification in Tweets",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "transformers>=4.0.0",
        "click",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "inference=inference.cli:main",
        ],
    },
    include_package_data=True,
)


