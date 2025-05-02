import numpy as np
import librosa
import joblib # To load the scaler
import tensorflow as tf
import os # For path checking
import io # To handle byte streams
import tempfile # To save uploaded file temporarily
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn 

# --- Configuration ---
# It's better practice to load these from environment variables or a config file,
# but we'll keep them here for simplicity based on the original script.
MODEL_LOAD_PATH = 'models/audio_distress_cnn_model.h5'
SCALER_LOAD_PATH = 'models/audio_feature_scaler.joblib'

# --- Global Variables for Model and Scaler ---
# These will be loaded during startup
model = None
scaler = None

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Audio Distress Prediction API",
    description="Upload an audio file to predict if it contains distress.",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Pydantic Response Model ---
class PredictionResponse(BaseModel):
    filename: str
    predicted_label: str
    probability_distress: float
    message: str = "Prediction successful"

# --- Feature Extraction Function (Identical to original script) ---
def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    try:
        # Load audio file. Using res_type='kaiser_fast' can speed up loading.
        X, sample_rate = librosa.load(file_name, sr=None, res_type='kaiser_fast') # Load original sample rate

        result = np.array([])
        stft = None # Compute STFT only if needed

        # Compute STFT if chroma or contrast is needed
        if chroma or contrast:
            # Using a smaller hop_length might capture finer details if needed, but increases computation
            stft = np.abs(librosa.stft(X)) # Use default hop_length (usually win_length // 4)

        # Extract features
        if mfcc:
            # n_mfcc=40 is common, ensure consistency with training
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            if stft is None: stft = np.abs(librosa.stft(X))
            # n_chroma=12 is standard
            chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_chroma=12).T, axis=0)
            result = np.hstack((result, chroma_features))
        if mel:
            # n_mels=128 is common, ensure consistency with training
            # fmax=sample_rate/2 or lower (e.g., 8000 Hz) can sometimes be useful
            mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128, fmax=8000).T, axis=0)
            result = np.hstack((result, mel_features))
        if contrast:
            if stft is None: stft = np.abs(librosa.stft(X))
            # n_bands=6 is default
            contrast_features = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate, n_bands=6).T, axis=0)
            result = np.hstack((result, contrast_features))
        if tonnetz:
            # Tonnetz is often computed on the harmonic component
            y_harmonic = librosa.effects.harmonic(X)
            tonnetz_features = np.mean(librosa.feature.tonnetz(y=y_harmonic, sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz_features))

        # print(f"Extracted features shape for {file_name}: {result.shape}") # Debug print
        return result

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        # Propagate the error for handling in the endpoint
        raise ValueError(f"Feature extraction failed for {file_name}: {e}")


# --- Startup Event Handler: Load Model and Scaler ---
@app.on_event("startup")
async def load_model_and_scaler():
    global model, scaler
    print("--- Server Startup: Loading Model and Scaler ---")

    # --- Basic File Checks ---
    if not os.path.exists(MODEL_LOAD_PATH):
        print(f"FATAL ERROR: Model file not found at {MODEL_LOAD_PATH}")
        # In a real app, you might want to prevent startup or use a default state
        raise FileNotFoundError(f"Model file not found at {MODEL_LOAD_PATH}")
    if not os.path.exists(SCALER_LOAD_PATH):
        print(f"FATAL ERROR: Scaler file not found at {SCALER_LOAD_PATH}")
        raise FileNotFoundError(f"Scaler file not found at {SCALER_LOAD_PATH}")

    try:
        model = tf.keras.models.load_model(MODEL_LOAD_PATH)
        print(f"Model loaded successfully from {MODEL_LOAD_PATH}")
        # Optional: Print model summary
        # model.summary()

        scaler = joblib.load(SCALER_LOAD_PATH)
        print(f"Scaler loaded successfully from {SCALER_LOAD_PATH}")
        print("--- Model and Scaler Loading Complete ---")

    except Exception as e:
        print(f"FATAL ERROR: Failed to load model or scaler during startup.")
        print(e)
        # Prevent the app from starting incorrectly
        raise RuntimeError(f"Failed to load resources: {e}")


# --- Prediction Endpoint ---
@app.post("/predict/", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(..., description="Audio file for distress prediction")):
    """
    Receives an audio file, extracts features, preprocesses them,
    and returns the distress prediction (label and probability).

    - **file**: The audio file (.wav, .mp3, etc.) to analyze.
    """
    global model, scaler

    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded. Server might be starting or encountered an error.")

    # --- Save Uploaded File Temporarily ---
    # librosa works best with file paths. We save the uploaded file content
    # to a temporary file.
    try:
        # Read the file content
        contents = await file.read()
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_audio_file:
            temp_audio_file.write(contents)
            temp_audio_path = temp_audio_file.name # Get the path to the temporary file
        print(f"Temporary audio file saved at: {temp_audio_path}")

        # --- Extract Features ---
        print(f"Extracting features from: {file.filename} (saved at {temp_audio_path})")
        # Use the same feature extraction settings as during training
        features = extract_feature(temp_audio_path, mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True)

        if features is None or features.size == 0:
            raise HTTPException(status_code=400, detail=f"Feature extraction failed for file: {file.filename}. The file might be corrupted or empty.")

        # --- Preprocess Features ---
        # 1. Reshape features to 2D for the scaler ([n_samples=1, n_features])
        features_2d = features.reshape(1, -1)

        # 2. Scale the features
        scaled_features = scaler.transform(features_2d)

        # 3. Reshape for Conv1D input ([n_samples=1, n_features, n_channels=1])
        reshaped_features = np.expand_dims(scaled_features, axis=-1)
        # print(f"Input shape for model: {reshaped_features.shape}") # Debug print

        # --- Make Prediction ---
        print("Making prediction...")
        prediction_proba = model.predict(reshaped_features)
        # The output is likely [[probability_of_class_1]] for binary classification
        distress_probability = float(prediction_proba[0][0]) # Ensure it's a standard float
        print(distress_probability)
        # --- Interpret Prediction ---
        predicted_class = (distress_probability > 0.5)
        label_map = {False: "No Distress", True: "Distress"}
        predicted_label = label_map[predicted_class]

        print(f"Prediction for {file.filename}: Label={predicted_label}, Probability={distress_probability:.4f}")

        # Return the result using the Pydantic model
        return PredictionResponse(
            filename=file.filename,
            predicted_label=predicted_label,
            probability_distress=distress_probability
        )

    except ValueError as ve: # Catch feature extraction errors specifically
        print(f"Feature extraction error: {ve}")
        raise HTTPException(status_code=400, detail=f"Error processing audio file {file.filename}: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        # Return a generic server error
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {e}")

    finally:
        # --- Clean up the temporary file ---
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                print(f"Temporary file {temp_audio_path} deleted.")
            except Exception as cleanup_error:
                print(f"Warning: Could not delete temporary file {temp_audio_path}: {cleanup_error}")
        # Ensure the uploaded file resource is closed (FastAPI might do this, but good practice)
        await file.close()


# --- Main Execution Block (for running with uvicorn) ---
if __name__ == '__main__':
    print("Starting FastAPI server...")
    # Check if model/scaler paths exist before starting server (optional but good)
    if not os.path.exists(MODEL_LOAD_PATH) or not os.path.exists(SCALER_LOAD_PATH):
        print("Error: Model or Scaler file not found. Please check paths.")
        print(f"Expected Model: {os.path.abspath(MODEL_LOAD_PATH)}")
        print(f"Expected Scaler: {os.path.abspath(SCALER_LOAD_PATH)}")
    else:
       uvicorn.run(app, host="0.0.0.0", port=8000)
       # Use "127.0.0.1" for host if you only want local access
       # uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) # Use reload for development