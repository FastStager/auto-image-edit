# AI Virtual Staging with SAM and Gemini

This application provides an interactive web interface to automatically detect furniture in a "staged" room image, segment it with high precision using contours, and then photorealistically place it into an "empty" room image using Google's Gemini AI.

### Key Features

* **Text-based Detection:** Use prompts like "sofa, lamp" to find objects.
* **Contour Segmentation:** Uses Segment Anything (SAM) for precise, non-rectangular object outlines.
* **Interactive Editor:** Click to select, reposition, and manage detected objects.
* **Two-Step AI Staging:** A robust workflow that first isolates furniture and then uses a powerful generative AI to ensure high-quality, realistic results.
* **Multi-View Workflow:** Use a generated image as a new input to create different arrangements or add more furniture.

## 1. Setup

This project uses `uv` for fast Python package and environment management.

**Prerequisites:**

* Python 3.9+
* Git

### Installation Steps

1. **Install `uv`** (if you don't have it):

    ```bash
    # On macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # On Windows
    irm https://astral.sh/uv/install.ps1 | iex
    ```

2. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/auto-imge-edit.git
    cd auto-image-edit
    ```

3. **Create and activate a virtual environment:**

    ```bash
    # Create the venv
    uv venv

    # Activate it
    # macOS / Linux
    source .venv/bin/activate
    # Windows
    .venv\Scripts\activate
    ```

4. **Install dependencies:**

    ```bash
    uv pip install -r requirements.txt
    ```

5. **Download the SAM Model:**
    The application will attempt to download this automatically on first run, but you can do it manually:

    ```bash
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    ```

6. **Set up Environment Variables:**
    The Enhanced AI Edit feature requires a Google Gemini API key.

    * Create a file named `.env` in the root of the project directory.
    * Add your API key to this file in the following format:

    ```env
    # .env
    GOOGLE_AI_STUDIO_API_KEY="your_api_key_here"
    ```

    > You can get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## 2. How to Run

With your virtual environment activated (`source .venv/bin/activate`), start the application with a single command:

```bash
python app.py
```

Open the local URL provided in your terminal (usually `http://127.0.0.1:7860`) in your web browser.

## 3. How to Use the App

The workflow is designed to be simple and powerful.

1. **Upload Images:** Provide an "Empty Room Image" and a "Staged Room Image".
2. **Detect Objects:** Enter text prompts (e.g., `sofa, chair, table`) and click **"Detect Objects"**. The detected items will be outlined with colored contours in the "Staged Room" tab.
3. **Transfer (Optional):** Click **"Transfer All to Empty Room"** to see how the objects will be positioned in the target room. You can adjust them here if needed.
4. **Prepare for AI:** Click **"Prepare for AI Edit"**. This removes the background from the staged image, leaving only the detected furniture with their colored borders. This is a crucial step for getting a clean result.
5. **Run AI Edit:** Click **"Run Enhanced AI Edit"**. The application sends the prepared furniture and the empty room to the Gemini API, which generates the final, photorealistic image.
6. **Multi-View (Optional):** If you like the result, click **"Use Generated as New Staged Image"** to use the output as a new starting point for further edits.
