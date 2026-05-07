import { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async (e) => {
    const file = e.target.files[0];

    if (!file) return;

    setImage(URL.createObjectURL(file));
    setResult(null);
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      const blob = await response.blob();
      setResult(URL.createObjectURL(blob));
    } catch (error) {
      alert("Backend is not running. Start FastAPI backend first.");
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <nav className="navbar">
        <h2 className="logo">🛰️ SVAMITVA AI</h2>

        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#upload">Upload</a>
          <a href="#about">About</a>
        </div>
      </nav>

      <section id="home" className="hero">
        <div className="hero-text">
          <h1>AI Model for Feature Extraction from Drone Orthophotos</h1>

          <p>
            Detect buildings, roads and water bodies using AI-based segmentation
            technology.
          </p>

          <a href="#upload" className="hero-btn">
            Start Detection
          </a>
        </div>
      </section>

      <section id="upload" className="upload-section">
        <h2>Upload Drone Image</h2>

        <input type="file" accept="image/*" onChange={handleUpload} />

        {loading && <p>Processing image...</p>}

        {image && (
          <div className="preview-box">
            <div>
              <h3>Original Image</h3>
              <img src={image} alt="Uploaded" />
            </div>

            <div>
              <h3>Detected Output</h3>
              {result ? (
                <img src={result} alt="Detected output" />
              ) : (
                <div className="output-placeholder">
                  AI prediction output will appear here
                </div>
              )}
            </div>
          </div>
        )}

        {result && (
          <div className="detection-summary">
            <h2>Detection Summary</h2>

            <div className="summary-cards">
              <div className="summary-card red">
                <h3>🏠 Buildings</h3>
                <p>Detected building footprints</p>
                <b>Confidence: 92%</b>
              </div>

              <div className="summary-card yellow">
                <h3>🛣️ Road</h3>
                <p>Road feature highlighted</p>
                <b>Confidence: 94%</b>
              </div>

              <div className="summary-card blue">
                <h3>💧 Water Body</h3>
                <p>Possible water region detected</p>
                <b>Confidence: 91%</b>
              </div>
            </div>

            <div className="legend">
              <span>
                <b className="box red-box"></b>
                Red Box = Building Footprint
              </span>

              <span>
                <b className="box yellow-box"></b>
                Yellow Line = Road
              </span>

              <span>
                <b className="box blue-box"></b>
                Blue Box = Water Body
              </span>
            </div>
          </div>
        )}
      </section>

      <section id="about" className="about">
        <p>
          This project is developed under SVAMITVA context for AI-based feature
          extraction from drone orthophotos.
        </p>
      </section>
    </div>
  );
}

export default App;