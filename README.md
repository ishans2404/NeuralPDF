# NeuralPDF: Interactive PDF Chat Web App

NeuralPDF is a web application that allows users to upload multiple PDF files and interact with their contents using AI-powered features. Users can engage in a chat conversation about the PDF content, generate quizzes with multiple-choice questions (MCQs), or create long-answer questions based on the uploaded documents. Access the live application [here](https://huggingface.co/spaces/ishans24/NeuralPDF).

## Features

- **Chat Conversation Mode**: Engage in a conversational chat about the PDF content.
- **Quiz & MCQs Mode**: Automatically generate multiple-choice quiz questions based on the PDF content.
- **Long-Answer Questions Mode**: Generate open-ended questions and answers based on the uploaded PDFs.
- **AI-Powered**: Uses advanced AI models for text understanding and generation.
- **Interactive Interface**: Built with Streamlit for a user-friendly experience.

## Installation

Ensure you have Python 3.9+ installed. Clone the repository and install the dependencies using `pip`:

```bash
git clone <repository-url>
cd NeuralPDF
pip install -r requirements.txt
```

## Running the Application

Get your Google API Key [here](https://ai.google.dev/aistudio):

```bash
export GOOGLE_API_KEY=<your secret key>
```
Run the Streamlit app using:

```bash
streamlit run app.py
```

The application will launch in your default web browser.

## Usage

1. **Upload PDF Files**: Upload one or more PDF files using the sidebar file uploader.
2. **Select Mode**: Choose between "Chat Conversation", "Quiz & MCQs", or "Long-Answer Questions" modes.
3. **Interact**: Depending on the mode selected, interact with the AI-generated responses or questions.
4. **Chat History**: Keep track of your conversation history within the app.
5. **Explore**: Experiment with different PDFs and modes to explore the capabilities of NeuralPDF.

## Dependencies

- `google-generativeai`
- `langchain_google_genai`
- `langchain`
- `langchain-community`
- `python-dotenv`
- `PyPDF2`
- `faiss-cpu`
- `streamlit`

Refer to `requirements.txt` for detailed dependencies.

## License

This project is licensed under the [Apache-2.0 license](https://github.com/ishans2404/NeuralPDF/blob/b38fde9589b9de8be8e56f11d7360d802afdc47e/LICENSE).
