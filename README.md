# Youtube Video Summarizer and Text-to-Speech

## Project Overview

This project aims to create a tool that can summarize YouTube videos and convert the summary to speech. Users can input a YouTube URL, and the application will return a concise summary of the video content along with an audio version of the summary.

## Target Audience

This project is particularly useful for:

- Students and researchers who need to quickly grasp the content of educational videos
- Professionals who want to efficiently review video content for work
- Content creators looking to generate summaries of their own or others' videos
- Anyone with visual impairments who prefers audio summaries of video content
- Individuals who want to save time by getting the key points of a video without watching it entirely

## Key Features

- YouTube video summarization
- Text-to-speech conversion of summaries
- User-friendly Gradio interface

## Technical Implementation

### Hugging Face Ecosystem

This project heavily utilizes the Hugging Face ecosystem, including:

- `transformers`: For accessing pre-trained models and fine-tuning
- `datasets`: For efficient data handling and processing
- `tokenizers`: For text tokenization

The implementation showcases the use of the Hugging Face Trainer API, which simplifies the process of fine-tuning and training models.

### Model

The project fine-tunes the "facebook/bart-large-cnn" model, which is well-suited for summarization tasks.

### Dataset

The model is fine-tuned on the How2 dataset, as described in the paper:

```
arXiv:1811.00347 "How2: A Large-scale Dataset for Multimodal Language Understanding"
```

This dataset consists of approximately 80,000 instructional videos (about 2,000 hours) with associated English subtitles and summaries.

### Data Processing

A key aspect of the implementation is the conversion of textual data to pandas DataFrames and then to the DictDataset format, which is compatible with the Hugging Face ecosystem. This demonstrates how to bridge traditional data processing techniques with modern NLP tools.

### Project Structure

The project is implemented in a modular, production-ready manner:

1. Data Ingestion
2. Data Transformation
3. Model Trainer

These components are combined in the `training_pipeline.py` file. The project also includes constants and entity files that allow for easy customization by developers.

### Fine-tuning

Due to computational constraints, the model was fine-tuned for only 1 epoch. Users with more resources may choose to increase this for potentially better results.

### User Interface

The project uses Gradio to create an intuitive interface for users to interact with the summarization and text-to-speech features.

### Text-to-Speech

The gTTS (Google Text-to-Speech) library is used to convert the generated summaries into speech.

## Getting Started

- Clone the repository
- Optional but recommended to create a conda/venv environment
- Use pip to install requirements using `pip install -r requirements.txt`
- run the app.py file
> The app on the first instance will take a few minutes since I haven't uploaded the model I finetuned in this repo due to its huge size, hence I have added an addional code for your usage in the comments in app file, the app will first download a model checkpoint from huggingface to run the app. This is only a one time initialization

## Future Improvements

- Fine-tune the model for more epochs to potentially improve performance
- Implement more advanced text-to-speech options
- Add support for multiple languages

## Contributing

We welcome contributions to improve and expand this project. Please feel free to submit issues and pull requests.

## Acknowledgements

- The authors of the How2 dataset (Please cite the following paper in all academic work that uses this dataset:)
```
@inproceedings{sanabria18how2,
  title = {{How2:} A Large-scale Dataset For Multimodal Language Understanding},
  author = {Sanabria, Ramon and Caglayan, Ozan and Palaskar, Shruti and Elliott, Desmond and Barrault, Lo\"ic and Specia, Lucia and Metze, Florian},
  booktitle = {Proceedings of the Workshop on Visually Grounded Interaction and Language (ViGIL)},
  year = {2018},
  organization={NeurIPS},
  url = {http://arxiv.org/abs/1811.00347}
}
```
