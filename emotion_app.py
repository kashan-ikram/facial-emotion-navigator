import gradio as gr
import torch
import numpy as np
import faiss
import pickle
from PIL import Image
from transformers import ViTForImageClassification, AutoImageProcessor
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

REPO_ID  = 'kashanikram/facial-emotion-vit'
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("Loading emotion model...")
processor = AutoImageProcessor.from_pretrained(REPO_ID)
model     = ViTForImageClassification.from_pretrained(REPO_ID)
model.eval()
device = torch.device('cpu')
model  = model.to(device)

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading RAG indexes...")
emotion_indexes = {}
emotion_chunks  = {}

for emotion in EMOTIONS:
    index_path = hf_hub_download(repo_id=REPO_ID, filename=f'faiss_{emotion}.bin', repo_type='model')
    emotion_indexes[emotion] = faiss.read_index(index_path)

chunks_path = hf_hub_download(repo_id=REPO_ID, filename='emotion_chunks.pkl', repo_type='model')
with open(chunks_path, 'rb') as f:
    emotion_chunks = pickle.load(f)

print("All models loaded!")

def detect_emotion(image):
    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs      = torch.softmax(outputs.logits, dim=1)[0]
    top_idx    = probs.argmax().item()
    emotion    = model.config.id2label[top_idx]
    confidence = probs[top_idx].item()
    top3 = sorted(enumerate(probs.tolist()), key=lambda x: x[1], reverse=True)[:3]
    top3_results = [(model.config.id2label[i], round(score*100, 1)) for i, score in top3]
    return emotion, round(confidence*100, 1), top3_results

def search_resources(emotion, top_k=3):
    query = f"I am feeling {emotion} and need support"
    query_embedding = embedder.encode([query]).astype('float32')
    index  = emotion_indexes[emotion]
    chunks = emotion_chunks[emotion]
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def agent_router(emotion, confidence):
    concerning = ['sad', 'angry', 'fear', 'disgust']
    positive   = ['happy']
    if confidence < 40:
        return 'uncertain'
    elif emotion in concerning:
        return 'support'
    elif emotion in positive:
        return 'positive'
    return 'neutral'

def run_agent(image):
    if image is None:
        return "Please upload a face image."
    try:
        pil_image = Image.fromarray(image)
        emotion, confidence, top3 = detect_emotion(pil_image)
        route     = agent_router(emotion, confidence)
        response  = f"Detected emotion: {emotion.upper()} ({confidence}%)\n\n"
        response += "Top predictions:\n"
        for e, c in top3:
            bar = '█' * int(c/5)
            response += f"  {e:10} {bar} {c}%\n"
        response += "\n"
        if route == 'uncertain':
            response += "Low confidence — please try with a clearer face image.\n\n"
        resources = search_resources(emotion, top_k=3)
        if route == 'support':
            response += "We noticed you might be going through something difficult.\nHere are some resources that may help:\n\n"
        elif route == 'positive':
            response += "Great to see you are feeling good! Here are some tips:\n\n"
        else:
            response += "Here are some mental wellness resources:\n\n"
        for i, resource in enumerate(resources, 1):
            response += f"{i}. {resource}\n\n"
        response += "---\nIf you are in crisis, call or text 9-8-8 (Canada) — available 24/7."
        return response
    except Exception as e:
        return f"Error: {str(e)}\nPlease try with a clear face image."

with gr.Blocks() as demo:
    gr.Markdown("# Facial Emotion Recognition + Mental Health Navigator")
    gr.Markdown("Upload a face image — AI detects emotion and provides relevant Canadian mental health resources. Powered by ViT + RAG + Agentic AI.")
    with gr.Row():
        image_input = gr.Image(label="Upload a face image", type="numpy")
        text_output = gr.Textbox(label="Emotion Analysis + Mental Health Resources", lines=20)
    submit_btn = gr.Button("Analyze")
    submit_btn.click(fn=run_agent, inputs=image_input, outputs=text_output)

if __name__ == "__main__":
    demo.launch()
