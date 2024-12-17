from setuptools import setup, find_packages
import os

# Read the content of your requirements.txt file
requirements = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt") as f:
        requirements = f.read().splitlines()

# Read the content of your README.md file
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Call the setup function
setup(
    name='deckdetect',  # package name
    version='1.0',  # package version
    description="Package deckdetect for the project",  # Short description of your package
    packages=find_packages(),  # Automatically find packages in the project directory
    install_requires=requirements,  # Dependencies listed in requirements.txt
    include_package_data=True,  # Include non-Python files like JSON, etc.
    url="https://github.com/JanReneHaak/deckdetect",  # Replace with your repository URL
)
