Ollama AI File Renamer
A powerful GUI-based utility that uses local Ollama models to intelligently rename files based on their content. It can analyze images, videos, PDFs, and text/code files to suggest descriptive, uniform filenames, helping you organize your digital life.

Features
Content-Aware Renaming: Leverages local multimodal and text-based LLMs through Ollama to understand the content of your files.

Broad File Support:

Images: Analyzes image content to create descriptive names (e.g., ginger_cat_sleeping_on_a_blue_armchair.jpg).

Videos: Extracts multiple frames, analyzes them, and summarizes the content for a relevant filename.

PDFs & Text: Reads text content from documents and code to suggest names based on their subject matter.

Smart Grouping: An optional second pass that uses an LLM to group similarly-themed files under a single, uniform name (e.g., kitten_on_couch_001.jpg, kitten_on_couch_002.jpg).

User-Friendly GUI: A simple Tkinter interface to select folders, choose models, and configure options without touching the command line.

Highly Customizable: All prompts sent to the AI models can be easily edited from within the application, allowing you to tailor the renaming logic to your exact needs.

Safe Operations: Includes a "Dry Run" mode by default to preview all proposed changes before any files are actually renamed.

Prerequisites
Before you begin, ensure you have the following installed and running:

Python 3.7+

Ollama: The application requires a running Ollama instance. You can download it from ollama.com.

Ollama Models: You need at least one vision model and one text model. You can pull them using the command line:

# Recommended vision model
ollama pull llava

# Recommended text model
ollama pull gemma:2b

FFmpeg: This is required for processing video files (extracting frames and reading duration).

Download FFmpeg from the official site: ffmpeg.org/download.html

Crucially, you must add the bin directory from your FFmpeg installation to your system's PATH environment variable so the script can execute ffmpeg and ffprobe.

Installation
Clone or Download the Repository:
Get the project files onto your local machine.

git clone https://github.com/your-username/ollama-file-renamer.git
cd ollama-file-renamer

Install Python Dependencies:
The required Python libraries are listed in requirements.txt. Install them using pip:

pip install -r requirements.txt

How to Use
Start the Application:
Make sure your Ollama application is running in the background. Then, run the gui.py script:

python gui.py

Select a Folder:
Click the "Browse..." button to choose the folder containing the files you want to rename.

Choose Your Models:

Select a Vision Model from the dropdown (e.g., llava:latest). This will be used for analyzing images and video frames.

Select a Text Model (e.g., gemma:2b). This is used for summarizing video frame descriptions, analyzing text/PDF files, and performing the optional grouping pass.

Click "Refresh Models" at any time to update the lists with your currently available Ollama models.

Set Your Options:

Dry Run: (Default: On) Preview all changes in the log without renaming any files. It is highly recommended to run this first.

Group Similar Filenames: (Default: Off) Enable the second AI pass to group files. This is slower but produces more uniform names for related files.

Auto-confirm all: (Default: Off) When unchecked, the script will not rename files even if "Dry Run" is off. Check this box to perform actual renaming operations. USE WITH CAUTION.

Adjust other settings like Temperature and Video Frames to Analyze as needed.

Run the Renamer:
Click the "Run Renamer" button to start the process. You can monitor the progress in the log output window. Click "Stop" at any time to gracefully halt the process.

Configuration
The application's behavior is controlled by the config.json file, which is created automatically on the first run.

models: Sets the default vision and text models to be selected on startup.

ollama_host: The URL for your Ollama instance.

generation_parameters: Controls the AI's creativity (temperature) and response length (num_predict).

video_processing: Sets the default number of frames to extract from videos.

prompts: Contains all the prompt templates sent to the language models. You can edit this file directly, but it's easier to use the "Edit Prompts..." button in the UI to fine-tune the AI's behavior.