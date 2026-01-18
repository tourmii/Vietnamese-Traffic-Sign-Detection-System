import os
import sys
from typing import List, Dict, Optional, Generator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sign_info_parser import (
    load_sign_regulations, 
    load_penalty_law, 
    load_classes,
    get_sign_info
)

from mistralai import Mistral


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# System prompt for the chatbot
SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên về biển báo giao thông Việt Nam và luật giao thông đường bộ.

Kiến thức của bạn bao gồm:
1. Quy chuẩn kỹ thuật quốc gia về báo hiệu đường bộ (QCVN 41:2019/BGTVT)
2. Nghị định 168/2024/NĐ-CP về xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ

Nhiệm vụ của bạn:
- Giải thích ý nghĩa các biển báo giao thông
- Trả lời câu hỏi về quy định giao thông
- Cung cấp thông tin về mức xử phạt vi phạm
- Tư vấn về cách ứng xử trong các tình huống giao thông
- Giải đáp thắc mắc về luật giao thông

Hướng dẫn:
- Trả lời bằng tiếng Việt
- Cung cấp thông tin chính xác, dễ hiểu
- Trích dẫn điều luật khi cần thiết
- Nếu không chắc chắn, hãy nói rõ và đề nghị tham khảo thêm nguồn chính thức

{context}"""


class TrafficSignChatbot:
    """Mistral-powered chatbot for traffic sign and law questions."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.client = None
        self.chat = None
        self.detected_signs_context = ""
        self.history = []
        self.model = "mistral-medium-latest"  # Model to use
        
        self._load_knowledge()
        
        if self.api_key:
            self._init_client()
    
    def _load_knowledge(self):
        """Load regulations and penalty documents."""
        try:
            self.regulations = load_sign_regulations()
            self.penalties = load_penalty_law()
            self.label_char, self.label_text = load_classes()
            print("Loaded traffic sign knowledge base")
        except Exception as e:
            print(f"Could not load knowledge base: {e}")
            self.regulations = ""
            self.penalties = ""
            self.label_char = {}
            self.label_text = {}
    
    def _init_client(self):
        """Initialize Mistral client."""
        try:
            self.client = Mistral(api_key=self.api_key)
            print("Mistral client initialized")
        except Exception as e:
            print(f"Could not initialize Mistral client: {e}")
            self.client = None
    
    def set_detected_signs(self, detections: List[Dict]):
        if not detections:
            self.detected_signs_context = ""
            return
        
        context_parts = ["Các biển báo đã phát hiện trong ảnh/video:"]
        seen_labels = set()
        
        for det in detections:
            label = det.get('label', 0)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            
            info = get_sign_info(label)
            if 'error' not in info:
                part = f"\n• Biển {info['sign_code']}: {info['vietnamese_name']}"
                part += f"\n  Phân loại: {info['category']}"
                if info.get('penalty_info'):
                    # Truncate penalty info
                    penalty = info['penalty_info'][:300]
                    part += f"\n  Mức phạt: {penalty}..."
                context_parts.append(part)
        
        self.detected_signs_context = "\n".join(context_parts)
    
    def _get_system_prompt(self) -> str:
        context = ""
        if self.detected_signs_context:
            context = f"\n\nNgữ cảnh hiện tại:\n{self.detected_signs_context}"
        
        return SYSTEM_PROMPT.format(context=context)
    
    def _search_knowledge(self, query: str) -> str:
        """Search knowledge base for relevant information."""
        import re
        
        relevant_info = []
        query_lower = query.lower()
        
        # Search for sign codes (e.g., P.102, W.201)
        sign_pattern = r'[PWRISa-z]\.?\d{3}[a-z]?'
        matches = re.findall(sign_pattern, query, re.IGNORECASE)
        
        for match in matches:
            # Search in regulations
            pattern = rf'{re.escape(match)}.*?(?=\n## |\n### |\Z)'
            reg_match = re.search(pattern, self.regulations, re.DOTALL | re.IGNORECASE)
            if reg_match:
                text = reg_match.group(0)[:500]
                relevant_info.append(f"Quy định về {match}:\n{text}")
        
        # Search for keywords
        keywords = ['cấm', 'phạt', 'tốc độ', 'điểm', 'biển', 'rẽ', 'quay đầu', 'dừng', 'đỗ']
        for kw in keywords:
            if kw in query_lower and kw in self.penalties.lower():
                # Find relevant section
                pattern = rf'Điều \d+\..*?{kw}.*?(?=\nĐiều |\Z)'
                matches = re.findall(pattern, self.penalties[:50000], re.DOTALL | re.IGNORECASE)
                for m in matches[:2]:  # Limit to 2 matches
                    if len(m) < 1000:
                        relevant_info.append(m)
        
        return "\n\n".join(relevant_info[:3])  # Limit to 3 sections
    
    def answer(self, question: str) -> str:
        """
        Answer a question about traffic signs or laws.
        
        Args:
            question: User's question
            
        Returns:
            Answer string
        """
        if not self.api_key:
            return "Chưa có API key. Vui lòng đặt biến môi trường MISTRAL_API_KEY hoặc truyền api_key khi khởi tạo."
        
        if not self.client:
            self._init_client()
            if not self.client:
                return "Không thể khởi tạo Mistral client. Vui lòng kiểm tra API key."
        
        try:
            # Search for relevant knowledge
            relevant_knowledge = self._search_knowledge(question)
            
            # Build system prompt
            system_prompt = self._get_system_prompt()
            
            # Add relevant knowledge to context
            if relevant_knowledge:
                system_prompt += f"\n\nThông tin liên quan từ văn bản pháp luật:\n{relevant_knowledge}"
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # Get response using Mistral API
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
            )
            
            answer = response.choices[0].message.content
            
            # Add to history
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
            
            return answer
            
        except Exception as e:
            return f"❌ Lỗi khi xử lý câu hỏi: {str(e)}"
    
    def stream_answer(self, question: str) -> Generator[str, None, None]:
        if not self.api_key:
            yield "❌ Chưa có API key."
            return
        
        if not self.client:
            self._init_client()
            if not self.client:
                yield "❌ Không thể khởi tạo Mistral client."
                return
        
        try:
            # Search for relevant knowledge
            relevant_knowledge = self._search_knowledge(question)
            
            # Build system prompt
            system_prompt = self._get_system_prompt()
            
            if relevant_knowledge:
                system_prompt += f"\n\nThông tin liên quan:\n{relevant_knowledge}"
            
            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            # Stream response using Mistral API
            response = self.client.chat.stream(
                model=self.model,
                messages=messages,
            )
            
            full_response = ""
            for chunk in response:
                if chunk.data.choices[0].delta.content:
                    full_response += chunk.data.choices[0].delta.content
                    yield full_response
            
            # Add to history
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            yield f"❌ Lỗi: {str(e)}"
    
    def reset_chat(self):
        """Reset chat history."""
        self.chat = None
        self.history = []
        self.detected_signs_context = ""
    
    def get_history(self) -> List[Dict]:
        """Get chat history."""
        return self.history


# Singleton instance
chatbot = TrafficSignChatbot()


def answer_question(question: str, detections: Optional[List[Dict]] = None) -> str:
    if detections:
        chatbot.set_detected_signs(detections)
    return chatbot.answer(question)


def reset_chatbot():
    chatbot.reset_chat()
