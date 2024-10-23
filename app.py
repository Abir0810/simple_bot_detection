from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pickle
import numpy as np

# Load the trained model
with open('bot_detection_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# Mount the static directory to serve the frontend HTML
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/frontend.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.post("/analyze")
async def analyze_user_data(request: Request):
    data = await request.json()
    
    # Extract features from the user interaction data
    mouse_speed = np.mean([abs(m['x']) + abs(m['y']) for m in data['mouseMovements']])
    keypress_interval = np.mean([k['time'] for k in data['keystrokes']])
    total_time = data['totalTime']
    
    # Create a feature array
    features = np.array([[mouse_speed, keypress_interval, total_time]])
    
    # Predict whether it's a human or a bot
    prediction = model.predict(features)
    
    return {"message": "You are a human!" if prediction[0] == 'human' else "Bot detected!"}