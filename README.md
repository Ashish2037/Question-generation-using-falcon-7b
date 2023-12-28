# Virtual Interviewer - Mira

## Overview
Virtual Interviewer Mira is an interactive Python application that uses speech recognition and natural language processing to simulate a data science interview. Mira asks randomly generated data science questions and evaluates user responses for similarity using sentence embeddings.

## Features
- Speech-to-text conversion for user responses
- Text-to-speech synthesis for providing instructions and questions
- Utilizes Hugging Face's model hub for question generation
- Measures similarity between user responses and expected answers using Sentence Transformers

# Usage
Run the check_similarity() function to start the virtual interview.
Listen to Mira's instructions and respond when prompted.
Mira will evaluate your responses based on cosine similarity to expected answers.

## Requirements
Make sure you have the required Python packages installed. You can install them using:

```bash
pip install -r requirements.txt


