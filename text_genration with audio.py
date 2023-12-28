import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
from langchain.llms import HuggingFaceHub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
from transformers import pipeline
from gtts import gTTS
import pygame
from io import BytesIO


def speech_to_text():
    
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Start Speeking:")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return 'No Audio'
        except sr.RequestError as e:
            return 'No Audio'


def text_to_audio(text):
    language = 'en'
    tts = gTTS(text=text, lang=language, slow=False)
    audio_stream = BytesIO()
    tts.write_to_fp(audio_stream)
    pygame.mixer.init()
    audio_stream.seek(0)
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
def check_similarity():

    text = "Hello,I am your virtual interviewer Mira. I am here to help you to enhance your data science skills. You can start speaking after I complete the question. Best of Luck!"
    text_to_audio(text)

    # Set Hugging Face Hub API token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Initialize Hugging Face Hub
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.6, "max_length": 100000, "max_new_tokens": 1000})

    # Generate questions
    query_result = llm('generate 2 very basic data science interview question, give only questions, questions are generated randomly, question should be diverse and based on basis')

    # Split the result into a list of questions
    query = query_result.split('\n')[1:]
    query = [question.split('. ', 1)[1] for question in query]
    

    flag = 0
    for i in range(len(query)):
        result = llm(query[i])[1:]
        print(f"Question: {query[i]}\n")
        text = query[i]
        text_to_audio(text)
        user_input = speech_to_text()
        print(user_input)
        embedding1 = model.encode(result, convert_to_tensor=True)
        embedding2 = model.encode(user_input, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embedding1, embedding2)
        if cosine_similarity.item() > 0.5:
            flag = flag + 1
        else:
            pass
    print(flag) 

check_similarity()