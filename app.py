import uvicorn
from fastapi import FastAPI, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Annotated
from analyzer import analyze_document
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware

# Initialize the FastAPI app
app = FastAPI(
    title="Document Authenticator API",
    description="Analyzes documents for forgery and extracts text using OCR.",
    version="1.0.0"
)

# --- Add the CORS Middleware ---
# This is the crucial part that allows your frontend (index.html) 
# to make requests to this backend server.
origins = [
    "*" # Allows all origins. For production, you might restrict this to your domain.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
# -----------------------------------


@app.get("/", tags=["General"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the Document Authenticator API!"}

@app.post("/analyze/", tags=["Analysis"])
async def analyze_image_endpoint(file: Annotated[bytes, File()]):
    """
    Receives an image, performs forgery detection and OCR,
    and returns the results.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    try:
        # Call the main analysis function from analyzer.py
        results = analyze_document(image_bytes=file)
        return JSONResponse(content=results, status_code=200)
    except Exception as e:
        # Catch any errors during analysis and return a server error
        raise HTTPException(status_code=500, detail=f"An error occurred during analysis: {str(e)}")

# This block allows you to run the app directly with `python app.py`
# For development, `uvicorn app:app --reload` is recommended.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
