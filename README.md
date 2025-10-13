⬡ Site Safety Intelligent Monitor (SSIM)
🧠 Building Safety with Intelligent Eyes
SSIM is an AI-powered real-time safety monitoring system for detecting PPE compliance and site hazards using computer vision and multimodal AI.
It integrates YOLOv11 for PPE detection, Gemini/LLaVA for hazard reasoning, and Streamlit for live monitoring — with Telegram alerts for violations.

🚀 Key Features
🦺 YOLOv11 PPE Detection – Identifies helmets, gloves, vests, shoes, etc.
⚠️ Violation Tracking – Detects and logs missing PPE with time thresholds.
🧠 VLM Analysis – Uses Gemini or LLaVA to detect environmental or behavioral hazards.
💬 Telegram Alerts – Sends instant photo/text alerts to a Telegram chat.
📊 Streamlit Dashboard – Live camera feed and real-time activity logs.
<img width="1660" height="877" alt="500189896-36bafbe6-0f73-4c17-8562-11050b8d300b" src="https://github.com/user-attachments/assets/23c9f3d3-3974-4900-88f6-bec0a96824d0" />
<img width="1024" height="1536" alt="20251009_1914_Construction Safety Alerts_remix_01k74vxpq9e8hs466hkgaqnvqx" src="https://github.com/user-attachments/assets/e4fcef62-e1d8-41cd-acb1-7367f4c4e2b9" />
🧰 Tech Stack
Component	Technology
Object Detection	Ultralytics YOLOv11
Vision-Language Model	Google Gemini / LLaVA
Dashboard	Streamlit
Alerts	Telegram Bot API
AI Frameworks	LangChain, langchain-google-genai
Backend	Python (asyncio, multiprocessing, OpenCV)

🧩 Future Plans
Multi-camera support
Multi-Agent Architecture for alert validation
Fine-tuning the VLM
Using a bigger YOLO model
