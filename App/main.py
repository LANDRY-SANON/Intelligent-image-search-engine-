from fastapi import FastAPI, Request, Form , File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import cv2
import numpy as np
from Engine_utils import search_similar_image_by_histogram, search_similar_images

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
templates = Jinja2Templates(directory=templates_dir)



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/get-feature-block/{feature_id}", response_class=HTMLResponse)
async def get_feature_block(request: Request, feature_id: int):
    if feature_id == 1:
        return templates.TemplateResponse("feature1.html", {"request": request})
    elif feature_id == 2:
        return templates.TemplateResponse("feature2.html", {"request": request})
    elif feature_id == 3:
        return templates.TemplateResponse("feature3.html", {"request": request})
    else:
        return "Invalid feature ID"

@app.post("/perform-feature-action")
async def perform_feature_action(request: Request, feature_id: int, queryImage: UploadFile = File(...), databaseFolderPath: str = Form(...)):
    if feature_id == 1:
        query_image_bytes = await queryImage.read()
        print(databaseFolderPath)
        # Save the query image to a temporary file
        query_image_path = "/tmp/query_image.jpg"
        with open(query_image_path, "wb") as query_image_file:
            query_image_file.write(query_image_bytes)

        # Perform search
        most_similar_image_path = search_similar_image_by_histogram(query_image_path, databaseFolderPath)
        
        # Return HTML with the search results
        #return templates.TemplateResponse("feature1_result.html", {"request": request, "most_similar_image_path": most_similar_image_path})