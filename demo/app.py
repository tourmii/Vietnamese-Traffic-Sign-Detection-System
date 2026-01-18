import os
import sys
import tempfile
import gradio as gr
from PIL import Image
import numpy as np

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.detector import TrafficSignDetector
from demo.chatbot import TrafficSignChatbot


# Initialize components
detector = TrafficSignDetector()
chatbot = TrafficSignChatbot()

# Track current detections for chatbot context
current_detections = []


def detect_image(image, model_type, confidence_threshold):
    """
    Handle image detection.
    
    Args:
        image: Input image (numpy array or file path)
        model_type: "YOLO" or "Faster R-CNN"
        confidence_threshold: Detection confidence threshold
        
    Returns:
        Tuple of (annotated_image, info_text)
    """
    global current_detections
    
    if image is None:
        return None, " Vui lòng tải lên một ảnh."
    
    try:
        # Save temp image if numpy array
        if isinstance(image, np.ndarray):
            temp_path = tempfile.mktemp(suffix='.jpg')
            Image.fromarray(image).save(temp_path)
            image_path = temp_path
        else:
            image_path = image
        
        # Map model type
        model_map = {"YOLO": "yolo", "Faster R-CNN": "faster_rcnn"}
        model = model_map.get(model_type, "yolo")
        
        # Detect
        annotated, detections, info_text = detector.detect_image(
            image_path, 
            model_type=model,
            threshold=confidence_threshold
        )
        
        # Update context for chatbot
        current_detections = detections
        chatbot.set_detected_signs(detections)
        
        # Convert BGR to RGB for display
        annotated_rgb = annotated[:, :, ::-1]
        
        return annotated_rgb, info_text
        
    except Exception as e:
        return None, f" Lỗi: {str(e)}"


def detect_video(video, model_type, confidence_threshold, progress=gr.Progress()):
    """
    Handle video detection.
    
    Args:
        video: Input video file path
        model_type: "YOLO" or "Faster R-CNN"
        confidence_threshold: Detection confidence threshold
        
    Returns:
        Tuple of (output_video_path, info_text)
    """
    global current_detections
    
    if video is None:
        return None, " Vui lòng tải lên một video."
    
    try:
        progress(0.1, desc="Đang xử lý video...")
        
        # Map model type
        model_map = {"YOLO": "yolo", "Faster R-CNN": "faster_rcnn"}
        model = model_map.get(model_type, "yolo")
        
        # Detect
        output_path, detections, info_text = detector.detect_video(
            video,
            model_type=model,
            threshold=confidence_threshold
        )
        
        progress(0.9, desc="Hoàn thành!")
        
        # Update context for chatbot
        current_detections = detections
        chatbot.set_detected_signs(detections)
        
        return output_path, info_text
        
    except Exception as e:
        return None, f" Lỗi: {str(e)}"


def chat_response(message, history):
    """
    Handle chat messages with streaming.
    
    Args:
        message: User message
        history: Chat history
        
    Yields:
        Partial responses for streaming
    """
    if not message.strip():
        yield ""
        return
    
    # Stream response
    for partial in chatbot.stream_answer(message):
        yield partial


def reset_chat():
    """Reset chatbot and clear detections context."""
    global current_detections
    current_detections = []
    chatbot.reset_chat()
    return [], " Đã đặt lại cuộc hội thoại."


# Custom CSS for styling
custom_css = """
/* Main body background */
body {
    background: linear-gradient(135deg, #0f1419 0%, #1a2332 25%, #0d2847 50%, #1a3a5c 75%, #0f1419 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh;
}

/* Main container */
.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    max-width: 1400px !important;
    margin: auto !important;
    background: transparent !important;
}

/* Dark mode wrapper */
.dark {
    background: transparent !important;
}

/* Blocks wrapper */
.contain {
    background: transparent !important;
}

/* Header styling */
.header-container {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #a855f7 100%);
    color: white;
    padding: 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(99, 102, 241, 0.4);
    border: 1px solid rgba(255,255,255,0.1);
}

.header-container h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.header-container p {
    font-size: 1.1rem;
    opacity: 0.9;
    margin: 0;
}

/* Tab styling */
.tab-nav {
    background: rgba(15, 23, 42, 0.8) !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 8px !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-bottom: none !important;
}

.tab-nav button {
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    color: #94a3b8 !important;
    background: transparent !important;
}

.tab-nav button:hover {
    background: rgba(99, 102, 241, 0.2) !important;
    color: white !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}

/* Tab panel styling */
.tabitem {
    background: rgba(15, 23, 42, 0.6) !important;
    border-radius: 0 0 16px 16px !important;
    padding: 1.5rem !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-top: none !important;
    backdrop-filter: blur(10px) !important;
}

/* Card styling */
.card {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    margin-bottom: 1rem;
    border: 1px solid rgba(99, 102, 241, 0.2);
}

/* Button styling */
.primary-btn {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    border-radius: 8px !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5) !important;
    background: linear-gradient(135deg, #0284c7, #4f46e5) !important;
}

/* Info text styling */
.info-text {
    background: linear-gradient(to bottom, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid #6366f1;
    max-height: 500px;
    overflow-y: auto;
    color: #e2e8f0 !important;
    border: 1px solid rgba(99, 102, 241, 0.2);
}

/* Chatbot container styling */
.chatbot-container {
    border-radius: 12px;
    overflow: hidden;
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
}

.chatbot-context-section {
    background: rgba(30, 41, 59, 0.9);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid rgba(99, 102, 241, 0.2);
}

/* Image/Video display */
.output-image, .output-video {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
}

/* Slider styling */
input[type="range"] {
    accent-color: #6366f1;
}

/* Dropdown styling */
.dropdown {
    border-radius: 8px !important;
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    color: #e2e8f0 !important;
}

/* Input components dark theme */
.gr-input, .gr-text-input, textarea, select, .gr-dropdown {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Labels */
label, .gr-text-label {
    color: #cbd5e1 !important;
}

/* Image upload area */
.gr-image, .gr-video {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 2px dashed rgba(99, 102, 241, 0.4) !important;
    border-radius: 12px !important;
}

.gr-image:hover, .gr-video:hover {
    border-color: rgba(99, 102, 241, 0.7) !important;
}

/* Row and column backgrounds */
.gr-row, .gr-column {
    background: transparent !important;
}

/* Chatbot messages */
.message {
    border-radius: 12px !important;
}

.message.user {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
}

.message.bot {
    background: rgba(51, 65, 85, 0.8) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
}

/* Contextual chatbot section header */
.context-chat-header {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15), rgba(99, 102, 241, 0.15));
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.context-chat-header h3 {
    color: #e2e8f0 !important;
    margin: 0 0 0.5rem 0;
}

.context-chat-header p {
    color: #94a3b8 !important;
    margin: 0;
    font-size: 0.9rem;
}
"""

# Build Gradio interface (Gradio 6.x compatible)
with gr.Blocks(title=" Vietnamese Traffic Sign Detection") as demo:
    
    # Header
    gr.HTML("""
        <div class="header-container">
            <h1> Hệ thống Nhận dạng Biển báo Giao thông Việt Nam</h1>
            <p>Phát hiện biển báo • Hiển thị thông tin chi tiết • Chatbot AI giải đáp luật giao thông</p>
        </div>
    """)
    
    with gr.Tabs() as tabs:
        
        # Tab 1: Image Detection with Contextual Chatbot
        with gr.Tab("📷 Phát hiện từ Ảnh", id="image_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label="Tải lên ảnh",
                        type="numpy",
                        sources=["upload", "clipboard"],
                        height=350
                    )
                    
                    with gr.Row():
                        img_model = gr.Dropdown(
                            choices=["YOLO", "Faster R-CNN"],
                            value="YOLO",
                            label="Chọn Model",
                            scale=1
                        )
                        img_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.95,
                            value=0.5,
                            step=0.05,
                            label="Ngưỡng tin cậy",
                            scale=1
                        )
                    
                    img_btn = gr.Button(
                        "🔍 Phát hiện biển báo",
                        variant="primary",
                        elem_classes=["primary-btn"]
                    )
                
                with gr.Column(scale=1):
                    img_output = gr.Image(
                        label="Kết quả phát hiện",
                        type="numpy",
                        height=350,
                        elem_classes=["output-image"]
                    )
            
            img_info = gr.Markdown(
                label="Thông tin biển báo",
                elem_classes=["info-text"]
            )
            
            # Contextual Chatbot Section - asks questions about detected signs
            gr.HTML("""
                <div class="context-chat-header">
                    <h3>💬 Hỏi đáp về biển báo đã phát hiện</h3>
                    <p>Đặt câu hỏi về các biển báo trong ảnh, quy định liên quan, hoặc mức phạt vi phạm.</p>
                </div>
            """)
            
            img_chatbot = gr.Chatbot(
                label="Hội thoại",
                height=400,
                elem_classes=["chatbot-container"]
            )
            
            with gr.Row():
                img_chat_input = gr.Textbox(
                    placeholder="Ví dụ: Nếu tôi vi phạm biển này thì bị phạt bao nhiêu?",
                    label="Câu hỏi của bạn",
                    scale=5
                )
                img_chat_btn = gr.Button(
                    "Gửi",
                    variant="primary",
                    scale=1,
                    elem_classes=["primary-btn"]
                )
            
            with gr.Row():
                gr.Examples(
                    examples=[
                        "Biển này có ý nghĩa gì?",
                        "Nếu tôi vi phạm biển này thì bị phạt bao nhiêu?",
                        "Khi thấy biển này tôi cần làm gì?",
                        "Giải thích quy định liên quan đến biển này."
                    ],
                    inputs=img_chat_input,
                    label="Câu hỏi mẫu"
                )
                img_chat_clear = gr.Button(" Xóa hội thoại", scale=1)
            
            def img_chat_respond(message, history):
                """Handle chat about detected signs in image."""
                if not message.strip():
                    return history, ""
                
                history = history or []
                history.append({"role": "user", "content": message})
                
                # Get response from chatbot (context is already set from detection)
                response = chatbot.answer(message)
                history.append({"role": "assistant", "content": response})
                
                return history, ""
            
            def clear_img_chat():
                """Clear image chatbot history."""
                chatbot.reset_chat()
                return [], ""
            
            img_btn.click(
                fn=detect_image,
                inputs=[img_input, img_model, img_threshold],
                outputs=[img_output, img_info]
            )
            
            img_chat_btn.click(
                fn=img_chat_respond,
                inputs=[img_chat_input, img_chatbot],
                outputs=[img_chatbot, img_chat_input]
            )
            
            img_chat_input.submit(
                fn=img_chat_respond,
                inputs=[img_chat_input, img_chatbot],
                outputs=[img_chatbot, img_chat_input]
            )
            
            img_chat_clear.click(
                fn=clear_img_chat,
                outputs=[img_chatbot, img_chat_input]
            )
        
        # Tab 2: Video Detection
        with gr.Tab("🎬 Phát hiện từ Video", id="video_tab"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input = gr.Video(
                        label="Tải lên video",
                        height=350
                    )
                    
                    with gr.Row():
                        vid_model = gr.Dropdown(
                            choices=["YOLO", "Faster R-CNN"],
                            value="YOLO",
                            label="Chọn Model",
                            scale=1
                        )
                        vid_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.95,
                            value=0.5,
                            step=0.05,
                            label="Ngưỡng tin cậy",
                            scale=1
                        )
                    
                    vid_btn = gr.Button(
                        "🔍 Phát hiện biển báo",
                        variant="primary",
                        elem_classes=["primary-btn"]
                    )
                
                with gr.Column(scale=1):
                    vid_output = gr.Video(
                        label="Kết quả phát hiện",
                        height=350,
                        elem_classes=["output-video"]
                    )
            
            vid_info = gr.Markdown(
                label="Thông tin biển báo",
                elem_classes=["info-text"]
            )
            
            # Contextual Chatbot Section for Video
            gr.HTML("""
                <div class="context-chat-header">
                    <h3>💬 Hỏi đáp về biển báo trong video</h3>
                    <p>Đặt câu hỏi về các biển báo đã phát hiện trong video, quy định liên quan, hoặc mức phạt vi phạm.</p>
                </div>
            """)
            
            vid_chatbot = gr.Chatbot(
                label="Hội thoại",
                height=300,
                elem_classes=["chatbot-container"]
            )
            
            with gr.Row():
                vid_chat_input = gr.Textbox(
                    placeholder="Ví dụ: Nếu tôi vi phạm biển này thì bị phạt bao nhiêu?",
                    label="Câu hỏi của bạn",
                    scale=5
                )
                vid_chat_btn = gr.Button(
                    "Gửi",
                    variant="primary",
                    scale=1,
                    elem_classes=["primary-btn"]
                )
            
            with gr.Row():
                gr.Examples(
                    examples=[
                        "Biển này có ý nghĩa gì?",
                        "Nếu tôi vi phạm biển này thì bị phạt bao nhiêu?",
                        "Khi thấy biển này tôi cần làm gì?",
                        "Giải thích quy định liên quan đến biển này."
                    ],
                    inputs=vid_chat_input,
                    label="Câu hỏi mẫu"
                )
                vid_chat_clear = gr.Button("🗑️ Xóa hội thoại", scale=1)
            
            def vid_chat_respond(message, history):
                """Handle chat about detected signs in video."""
                if not message.strip():
                    return history, ""
                
                history = history or []
                history.append({"role": "user", "content": message})
                
                # Get response from chatbot (context is already set from detection)
                response = chatbot.answer(message)
                history.append({"role": "assistant", "content": response})
                
                return history, ""
            
            def clear_vid_chat():
                """Clear video chatbot history."""
                chatbot.reset_chat()
                return [], ""
            
            vid_btn.click(
                fn=detect_video,
                inputs=[vid_input, vid_model, vid_threshold],
                outputs=[vid_output, vid_info]
            )
            
            vid_chat_btn.click(
                fn=vid_chat_respond,
                inputs=[vid_chat_input, vid_chatbot],
                outputs=[vid_chatbot, vid_chat_input]
            )
            
            vid_chat_input.submit(
                fn=vid_chat_respond,
                inputs=[vid_chat_input, vid_chatbot],
                outputs=[vid_chatbot, vid_chat_input]
            )
            
            vid_chat_clear.click(
                fn=clear_vid_chat,
                outputs=[vid_chatbot, vid_chat_input]
            )
        
        # Tab 3: General Chatbot (for questions not related to specific images)
        with gr.Tab("💬 Hỏi đáp chung", id="chat_tab"):
            gr.HTML("""
                <div class="context-chat-header">
                    <h3>🤖 Trợ lý AI Luật Giao thông</h3>
                    <p>Hỏi về biển báo giao thông, quy định, mức xử phạt, hoặc các tình huống giao thông cụ thể.</p>
                </div>
            """)
            
            general_chatbot = gr.Chatbot(
                label="Hội thoại",
                height=400,
                elem_classes=["chatbot-container"]
            )
            
            with gr.Row():
                general_chat_input = gr.Textbox(
                    placeholder="Ví dụ: Biển P.102 có ý nghĩa gì?",
                    label="Câu hỏi của bạn",
                    scale=5
                )
                general_chat_btn = gr.Button(
                    "Gửi",
                    variant="primary",
                    scale=1,
                    elem_classes=["primary-btn"]
                )
            
            with gr.Row():
                gr.Examples(
                    examples=[
                        "Biển P.102 có ý nghĩa gì?",
                        "Đi ngược chiều bị phạt bao nhiêu tiền?",
                        "Khi nào được quay đầu xe?",
                        "Giới hạn tốc độ trong khu dân cư?",
                        "Biển báo nguy hiểm có những loại nào?"
                    ],
                    inputs=general_chat_input,
                    label="Câu hỏi mẫu"
                )
                general_chat_clear = gr.Button("🗑️ Xóa hội thoại", scale=1)
            
            def general_chat_respond(message, history):
                """Handle general chat about traffic signs."""
                if not message.strip():
                    return history, ""
                
                history = history or []
                history.append({"role": "user", "content": message})
                
                # Clear any previous detection context for general chat
                chatbot.detected_signs_context = ""
                response = chatbot.answer(message)
                history.append({"role": "assistant", "content": response})
                
                return history, ""
            
            def clear_general_chat():
                """Clear general chatbot history."""
                chatbot.reset_chat()
                return [], ""
            
            general_chat_btn.click(
                fn=general_chat_respond,
                inputs=[general_chat_input, general_chatbot],
                outputs=[general_chatbot, general_chat_input]
            )
            
            general_chat_input.submit(
                fn=general_chat_respond,
                inputs=[general_chat_input, general_chatbot],
                outputs=[general_chatbot, general_chat_input]
            )
            
            general_chat_clear.click(
                fn=clear_general_chat,
                outputs=[general_chatbot, general_chat_input]
            )
    
    # Footer
    gr.HTML("""
        <div style="text-align: center; padding: 1.5rem; margin-top: 2rem; 
                    background: linear-gradient(to right, #f8fafc, #f1f5f9); 
                    border-radius: 12px;">
            <p style="color: #64748b; margin: 0;">
                 Vietnamese Traffic Sign Detection System | Powered by YOLO & Faster R-CNN + Gemini AI
            </p>
        </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        css=custom_css
    )
