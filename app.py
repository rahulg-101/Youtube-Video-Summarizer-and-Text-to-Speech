import os
import gradio as gr
import tempfile
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from YoutubeSummarizer.utils.main_utils import clean_text

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load models only once
model_path = os.path.join('artifacts', 'model_trainer')
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)  # use "facebook/bart-large-cnn" instead of model_path if you are cloning the repo since I haven't uploaded the pretrained model
summarization_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarization_pipe = pipeline('summarization', model=summarization_model, tokenizer=summarization_tokenizer)

tts_pipe = pipeline("text-to-speech", model="suno/bark-small")

def summarize_text(input_text):
    cleaned_text = clean_text(input_text)
    result = summarization_pipe(cleaned_text, max_length=150, min_length=50, do_sample=False)
    summary = result[0]['summary_text']
    return summary, "Summary generated! You can now use the 'Text-to-Speech' tab to convert this summary into speech."

# Define the text-to-speech function
def text_to_speech(sentence):
    tts = gTTS(sentence)
    tts.save("output.mp3")
    return "output.mp3"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Text Summarizer with Text-to-Speech")
    
    with gr.Tabs():
        with gr.TabItem("Summarization"):
            input_text = gr.Textbox(lines=10, label="Input Text")
            summarize_btn = gr.Button("Summarize")
            output_text = gr.Textbox(lines=5, label="Summary")
            info_text = gr.Textbox(label="Info", interactive=False)
            
            summarize_btn.click(
                fn=summarize_text,
                inputs=input_text,
                outputs=[output_text, info_text]
            )
        
        with gr.TabItem("Text-to-Speech"):
            gr.Markdown("## Copy the summary from the Summarization tab and paste it here to generate speech.")
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