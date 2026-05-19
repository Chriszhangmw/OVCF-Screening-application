# OVCF Clinical Web Demo

This repository contains a custom web demo for the OVCF multimodal screening system. The interface is designed for hospital demonstration scenarios and does not use the default Streamlit UI. It supports patient information entry, one-time image and video upload, real multimodal analysis, model prediction, and explainability display.

## Start the Demo

Run the following command from the project root:

```powershell
C:\Python313\python.exe -B code_final\web_demo\app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## How to Use

1. Enter the patient's basic clinical information.
2. Upload the required images or videos in one batch, such as front, back, side, left-turn, and right-turn views.
3. Keep **Enable real image/video analysis** selected, then click **Start Analysis**.
4. Before demonstrating the next patient, click **Clear and Start Next Patient**.

## Notes

- The DashScope API key is configured in the runtime script. If the `DASHSCOPE_API_KEY` environment variable is set, the environment variable takes priority.
- The model subprocess uses Anaconda Python by default and disables user-level `site-packages` to avoid NumPy and scikit-learn version conflicts.
- **Use cached demo results on failure** is only a fallback option. It is useful when the network or API is temporarily unavailable and the complete page effect still needs to be demonstrated.


