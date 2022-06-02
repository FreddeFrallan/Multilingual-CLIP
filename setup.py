from setuptools import setup, find_packages
from pathlib import Path
import os

if __name__ == "__main__":
    with Path(Path(__file__).parent, "README.md").open(encoding="utf-8") as file:
        long_description = file.read()

    def _read_reqs(relpath):
        fullpath = os.path.join(os.path.dirname(__file__), relpath)
        with open(fullpath) as f:
            return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]

    REQUIREMENTS = _read_reqs("requirements.txt")

    setup(
        name="multilingual_clip",
        packages=find_packages(),
        include_package_data=True,
        version="1.0.8",
        license="MIT",
        description="OpenAI CLIP text encoders for multiple languages!",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Fredrik Carlsson",
        author_email="FreddeFc@gmail.com",
        url="https://github.com/FreddeFrallan/Multilingual-CLIP",
        data_files=[(".", ["README.md"])],
        keywords=["machine learning"],
        install_requires=REQUIREMENTS,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
        ],
    )
