# Vietnamese text summarization
## Description 
This is a Python repo for vietnamese text summarization using the `vit5-large-vietnews-summarization` model develop by VietAI.

## Requirements 
- Python 3.9

## Installation 
Run the following command to install library 

`pip install -r requirements.txt`

to download the envit5 model file run the following command if the terminal is linux

`wget -P ./vit5_summary_model https://huggingface.co/VietAI/vit5-large-vietnews-summarization/resolve/main/pytorch_model.bin`

if the terminal is window, use the following command

`Invoke-WebRequest -Uri https://huggingface.co/VietAI/vit5-large-vietnews-summarization/resolve/main/pytorch_model.bin -OutFile ./vit5_summary_model/pytorch_model.bin`

## Usage
You can run the inference of the `vit5-large-vietnews-summarization` model with the `infer.py` file. And run the model API with the `vit5_summary_api.py` file

## Credits
This repo uses the vit5-large-vietnews-summarization model developed by VietAI, coupled with the Hugging Face Transformers library, which is an open-source library for NLP models.