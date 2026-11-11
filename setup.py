from setuptools import setup, find_packages

setup(
    name="csr_net",
    version="1.0.0",
    description="Contrastive Semantic Rectification and Confidence Calibration for Text Classification",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
    ],
)
