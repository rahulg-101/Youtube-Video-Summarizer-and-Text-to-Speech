# from YoutubeSummarizer.utils.main_utils import clean_text
# import os

# from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
# import gradio
# from IPython.display import Audio

# from transformers import pipeline


# def pipe_result(ipt):
    
#     model_path = os.path.join('artifacts\model_trainer')
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
#     tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

#     pipe = pipeline('summarization',model=model,tokenizer=tokenizer)

#     # ipt = input("Enter Text Here")
#     ipt = clean_text(ipt)
#     result = pipe(ipt)
#     return result[0]['summary_text']

# iface = gradio.Interface(fn=pipe_result,inputs=gradio.Textbox(lines=25),outputs=gradio.Textbox(lines=10))
# iface.launch()

# def audio_transcription():
#     pipe = pipeline("text-to-speech", model="suno/bark-small")
#     output = pipe(text)
#     Audio(output["audio"], rate=output["sampling_rate"])

import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from YoutubeSummarizer.utils.main_utils import clean_text
from IPython.display import Audio
import numpy as np
from scipy.io import wavfile

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load models only once
model_path = os.path.join('artifacts', 'model_trainer')
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
summarization_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarization_pipe = pipeline('summarization', model=summarization_model, tokenizer=summarization_tokenizer)

tts_pipe = pipeline("text-to-speech", model="suno/bark-small")



def summarize_text(input_text):
    cleaned_text = clean_text(input_text)
    result = summarization_pipe(cleaned_text)
    return result[0]['summary_text']

def text_to_speech(text):
    speech = tts_pipe(text)
    audio_array = np.array(speech['audio'])
    audio_array = audio_array / np.max(np.abs(audio_array))
    return (speech['sampling_rate'], audio_array)

def summarize_and_speak(input_text):
    summary = summarize_text(input_text)
    audio = text_to_speech(summary)
    
    # Save audio to a file
    output_file = "summary_audio.wav"
    wavfile.write(output_file, audio[0], (audio[1] * 32767).astype(np.int16))
    
    return summary, output_file

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Text Summarizer with Text-to-Speech")
    
    with gr.Row():
        input_text = gr.Textbox(lines=10, label="Input Text")
        output_text = gr.Textbox(lines=5, label="Summary")
    
    summarize_btn = gr.Button("Summarize")
    audio_output = gr.Audio(label="Summary Audio")
    
    summarize_btn.click(
        fn=summarize_and_speak,
        inputs=input_text,
        outputs=[output_text, audio_output]
    )

if __name__ == "__main__":
    demo.launch()