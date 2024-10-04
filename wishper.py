import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gradio as gr
from gtts import gTTS
import os
import tempfile


WHISPER_MODELS = [
    "./whisper-base",
    "./whisper-large-v3-turbo"
]

def load_model(model_id, use_gpu):
    device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if use_gpu and torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe

def transcribe_and_speak(audio, model_id, use_gpu, max_new_tokens, language):
    if audio is None:
        return "No audio detected. Please try again.", None
    
    pipe = load_model(model_id, use_gpu)
    
    result = pipe(audio, max_new_tokens=max_new_tokens, generate_kwargs={"language": language})
    transcribed_text = result["text"]
    
    tts = gTTS(text=transcribed_text, lang=language)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        output_audio_path = fp.name
    
    return transcribed_text, output_audio_path

iface = gr.Interface(
    fn=transcribe_and_speak,
    inputs=[
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Dropdown(choices=WHISPER_MODELS, value="./whisper-large-v3-turbo", label="Whisper Model"),
        gr.Checkbox(label="Use GPU (if available)"),
        gr.Slider(minimum=1, maximum=256, value=128, step=1, label="Max New Tokens"),
        gr.Dropdown(choices=["en", "fr", "de", "es", "it"], value="en", label="Language")
    ],
    outputs=[
        gr.Textbox(label="Transcribed Text"),
        gr.Audio(label="Text-to-Speech Output")
    ],
    title="Customizable Speech-to-Text-to-Speech with Whisper",
    description="Choose a Whisper model, set parameters, speak into the microphone, see the transcription, and hear it spoken back.",
)

# Launch the interface
iface.launch()