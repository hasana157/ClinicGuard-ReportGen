from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="clinicguard-reportgen",
    version="1.0.0",
    author="ClinicGuard ReportGen contributors",
    description="Evidence-grounded radiology report generation research prototype",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hasana157/ClinicGuard-ReportGen",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "med-train=scripts.train:main",
            "med-infer=scripts.inference:main",
            "med-eval=scripts.evaluate:main",
        ],
    },
)
