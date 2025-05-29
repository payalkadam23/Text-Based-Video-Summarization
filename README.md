# Text-Based-Video-Summarization
This repository contains the published research paper and implementation code for our Text Based Video Summarization approach. The framework leverages video frame extraction, object detection, and natural language processing to generate concise summaries based on user queries.

**Repository Structure**
research-project/
│
├── paper.pdf # Published research paper
├── code/
│ ├── colab_notebook.ipynb # Jupyter notebook to run the complete pipeline
│ ├── feature_extraction.py # Script for feature extraction
│ ├── frame_extraction.py # Script to extract video frames using OpenCV
│ ├── feature_list.pkl # Saved list of extracted features
├── input/
│ └── input_video.mp4 # Sample input video for testing
├── README.md # Project overview and usage instructions

 **Methodology Overview**
The proposed video summarization technique is broken into multiple steps to provide a comprehensive and personalized summary:
1.Frame Extraction
- Using OpenCV, video files are read and decomposed into individual frames.
- Images are resized to reduce computational cost while retaining key visual features.

 2.Object Detection
- We use EfficientDet, a state-of-the-art object detection model, to identify objects in the frames.
- TensorFlow is used for efficient numerical computations and deep learning operations.

3.Feature Extraction
- Features are extracted from detected objects and saved in a structured format (`feature_list.pkl`) for further processing.

 4. Query Processing with NLP
- User interest is captured via textual input.
- We use NLTK (Natural Language Toolkit) for:
  - Tokenization
  - Semantic analysis
  - Removal of stopwords
- This enables us to extract meaningful keywords and tailor the video summary based on user preferences.

5.  Video Summary Generation
- Based on relevant frames and object features matching user interest, a concise summary video is generated.



 **How to Run**

Option 1: Google Colab
1. Open `colab_notebook.ipynb` in [Google Colab](https://colab.research.google.com).
2. Upload the `input_video.mp4`.
3. Run each cell in sequence to:
   - Extract frames
   - Detect objects
   - Process user query
   - Generate summary

Option 2: Local Execution
Install required packages:
pip install -r requirements.txt
Then run:
python frame_extraction.py
python feature_extraction.py


**Citation**
Kadam, P., Vora, D., Patil, S., Mishra, S., & Khairnar, V. (2024). Behavioral profiling for adaptive video summarization: From generalization to personalization. MethodsX, 11, 102780. https://doi.org/10.1016/j.mex.2024.102780

