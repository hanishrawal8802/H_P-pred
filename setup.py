"""Setup configuration for the MLOps Churn Platform."""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="mlops-churn-platform",
    version="1.0.0",
    description="Production-grade MLOps platform for IBM Telco Customer Churn prediction",
    author="MLOps Team",
    python_requires=">=3.11",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "churn-train=training.trainer:main",
            "churn-serve=api.app:main",
            "churn-ingest=ingestion.data_loader:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
