from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from bigram_model import BigramModel

app = FastAPI()

# Sample corpus for the bigram model
# For a real application, a much larger corpus would be beneficial.
corpus = [
    "The king is powerful and rules the land",
    "The queen is wise and leads the people",
    "A cat sat on the mat",
    "A dog chased the cat"
]

# IMPORTANT: The server will take a moment to start because this line
# loads the large spaCy model into memory.
bigram_model = BigramModel(corpus)

# Pydantic model for the /generate endpoint
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

# Pydantic model for the new /similarity endpoint
class SimilarityRequest(BaseModel):
    text1: str
    text2: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    """
    Endpoint for the original text generation functionality.
    """
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    """
    New endpoint to calculate the semantic similarity between two texts.
    """
    similarity_score = bigram_model.get_similarity(request.text1, request.text2)
    return {
        "text1": request.text1,
        "text2": request.text2,
        "similarity_score": similarity_score
        }