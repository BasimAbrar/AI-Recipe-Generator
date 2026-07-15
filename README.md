# AI-Recipe-Generator
An AI-powered recipe recommendation system that detects ingredients from an uploaded image using a custom-trained YOLOv8 model and suggests the best matching recipes from 20,000+ Epicurious recipes. Built with Python, Streamlit, Ultralytics YOLOv8, and OpenCV.

The system uses a custom-trained YOLOv8 model together with a pretrained COCO model to identify fruits and vegetables, then matches the detected ingredients against more than 20,000 recipes from the Epicurious dataset. The application is built with Streamlit, allowing users to upload an image and instantly receive the top recipe recommendations.

## Features
- 🍎 Ingredient detection using YOLOv8
- 🤖 Dual-model detection (Custom + COCO)
- 📷 Image upload through Streamlit
- 🍽️ Recipe recommendation from 20,000+ recipes
- 📊 Confidence threshold adjustment
- ⚡ Fast inference with cached models

## Datasets

This project uses the following publicly available datasets:

- **LVIS Fruits & Vegetables Dataset** – Used to train the YOLOv8 ingredient detection model.
- **Epicurious Recipe Dataset** – Used for recipe matching and recommendations.

Due to licensing and repository size limitations, the datasets are not included in this repository. Please download them from their official sources and place them in the appropriate directories before running the project.

