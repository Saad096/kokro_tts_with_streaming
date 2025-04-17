# Kokoro TTS & Groq Streaming Voice Chat

This project demonstrates how to stream real-time audio responses using Groq's async API along with Kokoro TTS.

## Installation

1. **Install Dependencies**

```bash
   pip install coqui-tts
   pip install python-dotenv
   pip install groq
   pip install "fastrtc[vad, stt, tts]"
   pip install numpy
```

2. **Environment Setup**

* Create a file named .env in the root directory of the project.

* Add your Groq API key to the .env file as follows:
```env
GROQ_API_KEY=your_groq_api_key_here
```

3. **Virtual Environment:**

Create and activate a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate
```

4. **Run the Application:**

Launch the application by running:

```bash
python3 main.py
```


Open the provided local URL in your browser (e.g., http://127.0.0.1:7860) to use the application.