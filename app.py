import os
# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gradio as gr
import tempfile
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from YoutubeSummarizer.utils.main_utils import clean_text
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


# Load models only once
model_path = os.path.join('artifacts', 'model_trainer')
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)  # use "facebook/bart-large-cnn" instead of model_path if you are cloning the repo since I haven't uploaded the pretrained model
summarization_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarization_pipe = pipeline('summarization', model=summarization_model, tokenizer=summarization_tokenizer)

tts_pipe = pipeline("text-to-speech", model="suno/bark-small")

def extract_video_id(url):
    parsed_url = urlparse(url)
    video_id = parse_qs(parsed_url.query).get('v')
    if video_id:
        return video_id[0]
    else:
        path = parsed_url.path.split('/')
        if path[-1]:
            return path[-1]
        elif len(path) > 1:
            return path[-2]
    return None

def transcript(video_id):
    Transc = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join(i['text'] for i in Transc)

def chunk_text(text, max_chunk_size=1000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    for word in words:
        if current_size + len(word) + 1 > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_text(input_text):
    video_id = extract_video_id(input_text)
    if not video_id:
        return "Invalid YouTube URL", "Please provide a valid YouTube URL"
    
    print(f"Video Id : {video_id}")
    input_text = transcript(video_id)
    print(f"Transcript length: {len(input_text)} characters")
    
    cleaned_text = clean_text(input_text)
    chunks = chunk_text(cleaned_text)
    
    summaries = []
    for chunk in chunks:
        result = summarization_pipe(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(result[0]['summary_text'])
    
    final_summary = " ".join(summaries)
    
    # If the combined summary is still too long, summarize it again
    if len(final_summary) > 1000:
        final_summary = summarization_pipe(final_summary, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    
    return final_summary, "Summary generated! You can now use the 'Text-to-Speech' tab to convert this summary into speech."

# Define the text-to-speech function
def text_to_speech(sentence):
    tts = gTTS(sentence)
    tts.save("output.mp3")
    return "output.mp3"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>Youtube How-to Videos Text Summarizer with Text-to-Speech</h1>")
    gr.Markdown("This application is specifically fine-tuned to generate summaries of 'How-to' instructional videos. Currently, it only supports videos with English transcripts and may produce inappropriate summaries, as I have fine-tuned the model for just one epoch due to limited computational resources. Additionally, you can create audio speech for the generated summaries in the Text-to-Speech tab.")
    with gr.Tabs():
        
        with gr.TabItem("Summarization"):
            input_text = gr.Textbox(label="Input Youtube URL")
            summarize_btn = gr.Button("Summarize")
            output_text = gr.Textbox(lines=5, label="Summary")
            info_text = gr.Textbox(label="Info", interactive=False)
            
            summarize_btn.click(
                fn=summarize_text,
                inputs=input_text,
                outputs=[output_text, info_text]
            )
        
        with gr.TabItem("Text-to-Speech"):
            
            gr.Markdown("### Copy the summary from the Summarization tab and paste it here to generate speech.")
            tts_input = gr.Textbox(lines=5, label="Text to Convert")
            tts_btn = gr.Button("Generate Speech")
            audio_output = gr.Audio(label="Generated Speech")
            
            tts_btn.click(
                fn=text_to_speech,
                inputs=tts_input,
                outputs=audio_output
            )

if __name__ == "__main__":
    demo.launch()