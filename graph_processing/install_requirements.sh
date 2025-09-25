#!/bin/bash

echo "Installing requirements for Vision Pipeline..."

# Core dependencies
pip3 install --user PyMuPDF  # fitz
pip3 install --user pillow
pip3 install --user torch torchvision
pip3 install --user ultralytics
pip3 install --user openai
pip3 install --user python-dotenv
pip3 install --user pydantic
pip3 install --user pandas
pip3 install --user tqdm

echo "Installation complete!"