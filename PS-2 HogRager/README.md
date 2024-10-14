# Hog RAGger 

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)

## Overview

The Streamlit app serves as an interactive interface for users to explore and utilize machine learning models, specifically designed for natural language processing tasks. This application leverages libraries such as LangChain and Hugging Face for embeddings and various functionalities.

## Installation

To set up the application, follow these steps:

1. Clone this repository:
    ```bash
    git clone https://github.com/shrey7ansh07/InterIIT.git
    cd InterIIT
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit app, use the following command:

```bash
streamlit run app.py
```

## Requirements

The following libraries are required to run the application. All these dependencies can be installed via `requirements.txt`:

- `langchain`
- `huggingface_hub`
- `tiktoken`
- `ctransformers`
- `accelerate`
- `sentence_transformers`
- `faiss-cpu`
- `InstructorEmbedding`
- `langchain-community`
- `chromadb`
- `pysqlite3`

Make sure to have Python 3.7 or higher installed on your machine to avoid compatibility issues.



