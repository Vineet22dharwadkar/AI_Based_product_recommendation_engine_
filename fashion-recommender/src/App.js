import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
  const [randomImages, setRandomImages] = useState([]);
  const [similarImages, setSimilarImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('resnet50');
  const [predictions, setPredictions] = useState({});

  useEffect(() => {
    fetchRandomImages();
  }, []);

  const fetchRandomImages = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:5000/random_images');
      setRandomImages(response.data);
    } catch (error) {
      console.error('Error fetching random images:', error);
    }
  };

  const fetchSimilarImages = async (imagePath) => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/similar_images', { image: imagePath, model: selectedModel });
      setSimilarImages(response.data);
      setSelectedImage(imagePath);
    } catch (error) {
      console.error('Error fetching similar images:', error);
    }
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleFileUpload = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model', selectedModel);
    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setSimilarImages(response.data);
      setSelectedImage(URL.createObjectURL(selectedFile));
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPredictions(response.data);
    } catch (error) {
      console.error('Error making predictions:', error);
    }
  };

  return (
    <div className="App">
      <header className="Header">
        <h1>Fashion Recommender System</h1>
        <div className="search-container">
          <select value={selectedModel} onChange={handleModelChange}>
            <option value="resnet50">ResNet50</option>
            <option value="vgg16">VGG16</option>
            <option value="inceptionv3">InceptionV3</option>
            <option value="efficientnet">EfficientNet</option>
            <option value="knn">KNN</option>
            <option value="svm">SVM</option>
            <option value="random_forest">Random Forest</option>
            <option value="decision_tree">Decision Tree</option>
          </select>
          <input type="file" onChange={handleFileChange} />
          <button onClick={handleFileUpload}>Upload and Find Similar Images</button>
          <button onClick={handlePredict}>Predict</button>
        </div>
      </header>

      <main>
        {selectedImage ? (
          <div>
            <div className="selected-image-container">
              <img src={selectedImage} alt="Selected" className="large-image" />
            </div>
            <h2>Similar Images</h2>
            <div className="image-grid">
              {similarImages.map((img, idx) => (
                <img
                  key={idx}
                  src={`http://127.0.0.1:5000/${img}`}
                  alt={`Similar ${idx}`}
                  className="image"
                />
              ))}
            </div>
            <button onClick={() => setSelectedImage(null)}>Back to Random Images</button>
          </div>
        ) : (
          <div>
            <h2>Random Images</h2>
            <div className="image-grid">
              {randomImages.map((img, idx) => (
                <img
                  key={idx}
                  src={`http://127.0.0.1:5000/uploads/${img}`}
                  alt={`Random ${idx}`}
                  className="image"
                  onClick={() => fetchSimilarImages(img)}
                />
              ))}
            </div>
          </div>
        )}
        {predictions && Object.keys(predictions).length > 0 && (
          <div>
            <h2>Predictions</h2>
            <ul>
              {Object.entries(predictions).map(([model, prediction], idx) => (
                <li key={idx}>{model}: {prediction}</li>
              ))}
            </ul>
          </div>
        )}
      </main>

      <footer>
        <p>&copy; 2024 Fashion Recommender System</p>
      </footer>
    </div>
  );
};

export default App;
