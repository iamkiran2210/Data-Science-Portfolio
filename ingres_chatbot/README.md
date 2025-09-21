# 💧 INGRES AI ChatBot

## Overview

This repository contains the source code for the **INGRES AI ChatBot**, a prototype developed for the **Smart India Hackathon 2025** (Problem Statement ID: SIH25066).

The application provides an intuitive, conversational interface for farmers, government officials, and researchers to access India's groundwater data. It is built with a scalable architecture using Python, Streamlit, and local AI models via Ollama to ensure data sovereignty and offline capability.

## Key Features

- **Hybrid Intelligence (MCP):** The chatbot intelligently routes user queries, either fetching structured data from the mock INGRES database or consulting a general-purpose local LLM.
- **Local AI Integration:** Runs on local hardware using Ollama (`qwen2:7b` model) for maximum data security and offline functionality.
- **Rich Data Presentation:** Responds to data queries with text summaries, interactive Folium maps, and Plotly charts.
- **Full Accessibility:** Includes full Voice I/O (speech-to-text and text-to-speech).
- **Multilingual Support:** The chatbot can understand and respond in English, Hindi, and Telugu.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python 3.8+**
2.  **Ollama Service:** The application requires the Ollama service to be running in the background. You can download it from the [official Ollama website](https://ollama.com/).
3.  **PortAudio:** This is a system-level dependency required for the voice features.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install System Dependencies (for Debian/Ubuntu):**
    Open a terminal and run the following command to install PortAudio.
    ```bash
    sudo apt-get update && sudo apt-get install -y portaudio19-dev
    ```

3.  **Install Python Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r ingres_chatbot/requirements.txt
    ```

4.  **Download the AI Model:**
    The application will download the required model on first run, but you can also pull it manually beforehand.
    ```bash
    ollama pull qwen2:7b
    ```

## Running the Application

1.  **Ensure the Ollama service is running.** In most installations, it runs automatically as a background service. You can check its status with `systemctl status ollama`.

2.  **Run the Streamlit app:**
    From the root directory of the repository, run the following command:
    ```bash
    streamlit run ingres_chatbot/app.py
    ```

3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
