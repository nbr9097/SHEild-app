import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios'; // Using axios for easier request handling
import './Home.css';
// Assuming sheildLogo is used somewhere, otherwise remove it
// import sheildLogo from '../assets/sheildLogo.png';
import homepageImage from '../assets/homepageImage.png';

const Home = () => {
  // --- State Variables ---
  const [selectedFile, setSelectedFile] = useState(null); // Holds the File object (uploaded or recorded)
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null); // For playback preview
  const [predictionResult, setPredictionResult] = useState(null); // { filename, predicted_label, probability_distress }
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // --- Refs ---
  const fileInputRef = useRef(null); // To trigger file input click
  const mediaRecorderRef = useRef(null); // To hold MediaRecorder instance
  const audioChunksRef = useRef([]); // To store recorded audio chunks

  // --- Constants ---
  // Make sure this URL points to your running FastAPI backend
  const PREDICT_API_URL = 'http://localhost:8000/predict/'; // Adjust port if needed

  // --- Cleanup Audio URL ---
  // Revoke object URL to free memory when component unmounts or audio changes
  useEffect(() => {
    return () => {
      if (audioURL) {
        URL.revokeObjectURL(audioURL);
      }
    };
  }, [audioURL]);

  // --- Event Handlers ---

  // 1. Handle File Selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPredictionResult(null); // Clear previous results
      setError(null);
      // Create a URL for preview
      const url = URL.createObjectURL(file);
      setAudioURL(url);
    }
    // Reset the file input value so the same file can be selected again
    event.target.value = null;
  };

  // 2. Trigger File Input
  const handleUploadClick = () => {
    // Clear any previous recording state/data
    stopRecordingCleanup(); // Ensure recorder is stopped if active
    setSelectedFile(null);
    setAudioURL(null);
    setPredictionResult(null);
    setError(null);
    // Trigger the hidden file input
    fileInputRef.current.click();
  };

  // 3. Start/Stop Recording
  const handleRecordClick = async () => {
    if (isRecording) {
      // --- Stop Recording ---
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
        mediaRecorderRef.current.stop(); // This triggers the 'onstop' event
      }
      setIsRecording(false);
       // Note: File creation happens in the 'onstop' handler
    } else {
       // --- Start Recording ---
      setSelectedFile(null); // Clear previous file selection
      setAudioURL(null);
      setPredictionResult(null);
      setError(null);

      try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Check browser support for preferred MIME type (optional, adjust as needed)
        // Common types: 'audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/wav'
        const options = { mimeType: 'audio/wav' }; // Try WAV first
        let recorder;
        try {
            recorder = new MediaRecorder(stream, options);
        } catch (e) {
            console.warn("WAV mimeType not supported, trying default");
            try {
                recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' }); // Common fallback
            } catch (e2) {
                 console.warn("WebM/Opus mimeType not supported, trying browser default");
                 recorder = new MediaRecorder(stream); // Browser default
            }
        }

        mediaRecorderRef.current = recorder;
        audioChunksRef.current = []; // Clear previous chunks

        // Collect audio data chunks
        mediaRecorderRef.current.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        // When recording stops, create the Blob and File
        mediaRecorderRef.current.onstop = () => {
          const mimeType = mediaRecorderRef.current.mimeType || 'audio/wav'; // Get actual mime type used
          const fileExtension = mimeType.split('/')[1].split(';')[0]; // e.g., 'wav' or 'webm'
          const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
          const recordedFile = new File([audioBlob], `recorded_audio.${fileExtension}`, { type: mimeType });

          setSelectedFile(recordedFile);
          const url = URL.createObjectURL(recordedFile);
          setAudioURL(url);

          // Clean up stream tracks
          stream.getTracks().forEach(track => track.stop());
        };

        // Start recording
        mediaRecorderRef.current.start();
        setIsRecording(true);

      } catch (err) {
        console.error("Error accessing microphone:", err);
        setError(`Error accessing microphone: ${err.message}. Please grant permission.`);
        setIsRecording(false);
      }
    }
  };

 // Helper to stop recording and clean up resources
  const stopRecordingCleanup = () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
          mediaRecorderRef.current.stop();
      }
      // Also stop tracks if the stream is still active (e.g., user clicks Upload while recording)
      if (mediaRecorderRef.current && mediaRecorderRef.current.stream) {
          mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      }
      setIsRecording(false);
      mediaRecorderRef.current = null;
      audioChunksRef.current = [];
  };


  // 4. Handle Prediction Request
  const handlePredictClick = async () => {
    if (!selectedFile) {
      setError("Please upload or record audio first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setPredictionResult(null);

    const formData = new FormData();
    // The backend expects the file under the key 'file'
    formData.append('file', selectedFile, selectedFile.name);

    try {
      const response = await axios.post(PREDICT_API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data', // Axios might set this automatically for FormData, but being explicit is fine
        },
      });
      console.log("Prediction Response:", response.data);
      setPredictionResult(response.data);
    } catch (err) {
      console.error("Prediction Error:", err);
      let errorMsg = "Prediction failed. Please try again.";
      if (err.response) {
        // Server responded with a status code outside the 2xx range
        console.error("Error data:", err.response.data);
        console.error("Error status:", err.response.status);
        errorMsg = `Prediction failed: ${err.response.data.detail || err.response.statusText} (Status: ${err.response.status})`;
      } else if (err.request) {
        // Request was made but no response received (e.g., network error, server down)
        console.error("Error request:", err.request);
        errorMsg = "Prediction failed: Could not connect to the server. Is it running?";
      } else {
        // Something else happened in setting up the request
        errorMsg = `Prediction failed: ${err.message}`;
      }
      setError(errorMsg);
      setPredictionResult(null); // Clear any potential stale results
    } finally {
      setIsLoading(false);
    }
  };

  // --- Render Component ---
  return (
    <div> {/* Removed redundant outer div */}
      <div className="home-container">
        <div className="left-section">
          <h1>
            WELCOME TO <span className="highlight">SHEild</span>
          </h1>

          {/* Hidden File Input */}
          <input
            type="file"
            accept="audio/*" // Accept any audio format
            ref={fileInputRef}
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />

          {/* Action Buttons */}
          <div className="action-buttons">
            <button
                className="upload-btn"
                onClick={handleUploadClick}
                disabled={isRecording || isLoading} // Disable if recording or loading
            >
                Upload Audio
            </button>
            {/* <button
                className={`record-btn ${isRecording ? 'recording' : ''}`}
                onClick={handleRecordClick}
                disabled={isLoading} // Disable only if loading
            >
                {isRecording ? 'Stop Recording' : 'Record Audio'}
            </button> */}
            <button
                className="predict-btn"
                onClick={handlePredictClick}
                disabled={!selectedFile || isLoading || isRecording} // Disable if no file, loading, or recording
            >
                Predict Distress
            </button>
          </div>

          {/* File Info & Preview */}
          {selectedFile && (
            <div className="file-info">
              <p>Selected: {selectedFile.name}</p>
              {audioURL && <audio controls src={audioURL} />}
            </div>
          )}

          {/* Loading Indicator */}
          {isLoading && <p className="status-loading">Predicting...</p>}

          {/* Error Display */}
          {error && <p className="status-error">Error: {error}</p>}

          {/* Prediction Result */}
          {predictionResult && (
            <div className="prediction-result">
              <h3>Prediction Result:</h3>
              <p>File: {predictionResult.filename}</p>
              <p>
                Label: <span className={`label-${predictionResult.predicted_label.toLowerCase().replace(' ', '-')}`}>
                           {predictionResult.predicted_label}
                       </span>
              </p>
              <p>Probability (Distress): {(predictionResult.probability_distress * 100).toFixed(2)}%</p>
            </div>
          )}

        </div>
        <div className="right-section">
          <img src={homepageImage} alt="Voice Illustration" className="home-image" />
        </div>
      </div>
    </div>
  );
};

export default Home;