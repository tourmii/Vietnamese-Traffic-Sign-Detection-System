import os
import re
from typing import Optional


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_classes():
    classes_char_file = os.path.join(PROJECT_ROOT, "utils/classes.txt")
    classes_vie_file = os.path.join(PROJECT_ROOT, "utils/classes_vie.txt")
    
    with open(classes_char_file, 'r', encoding='utf-8') as f:
        classes_char = [line.strip() for line in f.readlines()]
    
    with open(classes_vie_file, 'r', encoding='utf-8') as f:
        classes_vie = [line.strip() for line in f.readlines()]
    
    LABEL_CHAR = {i+1: label for i, label in enumerate(classes_char)}
    LABEL_TEXT = {i+1: label for i, label in enumerate(classes_vie)}
    
    return LABEL_CHAR, LABEL_TEXT


def load_sign_regulations():
    file_path = os.path.join(PROJECT_ROOT, "text_data/51_bgtvt_kt.md")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_penalty_law():
    file_path = os.path.join(PROJECT_ROOT, "text_data/168_2024_NĐ_CP.md")
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_sign_code(label_char: str) -> str:
    base_code = label_char.split('*')[0]
    return base_code


def get_sign_category(sign_code: str) -> str:
    if sign_code.startswith('P.'):
        return "Biển báo cấm"
    elif sign_code.startswith('W.'):
        return "Biển báo nguy hiểm"
    elif sign_code.startswith('R.'):
        return "Biển báo hiệu lệnh"
    elif sign_code.startswith('I.') or sign_code.startswith('S.'):
        return "Biển chỉ dẫn"
    else:
        return "Biển báo khác"


def search_sign_in_regulations(sign_code: str, regulations_text: str) -> Optional[str]:
    base_code = extract_sign_code(sign_code)
    
    patterns = [
        rf'## B\.\d+[a-z]?\s+Biển số {re.escape(base_code)}.*?(?=\n## B\.|\nPhụ lục|\Z)',
        rf'### B\.\d+[a-z]?\s+Biển số {re.escape(base_code)}.*?(?=\n## B\.|\n### B\.|\nPhụ lục|\Z)',
        rf'B\.\d+[a-z]?\s+Biển số {re.escape(base_code)}.*?(?=\n## B\.|\n### B\.|\nPhụ lục|\Z)',
        rf'Biển số {re.escape(base_code)}.*?(?=\n## |\n### |\nPhụ lục|\Z)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, regulations_text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(0).strip()
            if len(text) > 2000:
                text = text[:2000] + "\n..."
            return text
    
    return None


def get_related_penalties(sign_code: str, vietnamese_name: str, penalty_text: str) -> Optional[str]:
    base_code = extract_sign_code(sign_code)
    category = get_sign_category(sign_code)
    
    penalties = []
    
    keyword_map = {
        'P.102': ['đi ngược chiều', 'ngược chiều'],
        'P.103': ['cấm ô tô', 'xe ô tô'],
        'P.104': ['cấm xe máy', 'xe mô tô', 'xe gắn máy'],
        'P.106': ['cấm xe tải', 'xe ô tô tải'],
        'P.107': ['cấm ô tô khách', 'xe ô tô khách'],
        'P.108': ['rơ-moóc', 'rơ moóc', 'moóc'],
        'P.117': ['chiều cao', 'hạn chế chiều cao'],
        'P.123': ['cấm rẽ trái', 'cấm rẽ phải', 'rẽ trái', 'rẽ phải'],
        'P.124': ['quay đầu', 'cấm quay đầu'],
        'P.127': ['tốc độ', 'quá tốc độ', 'chạy quá tốc độ'],
        'P.130': ['dừng xe', 'đỗ xe', 'cấm dừng', 'cấm đỗ'],
        'P.131': ['cấm đỗ xe', 'đỗ xe'],
        'W.': ['biển báo nguy hiểm', 'cảnh báo'],
        'R.': ['hiệu lệnh', 'chỉ dẫn'],
    }
    
    keywords = []
    for code_prefix, kw_list in keyword_map.items():
        if base_code.startswith(code_prefix):
            keywords.extend(kw_list)
    
    if vietnamese_name:
        keywords.append(vietnamese_name.lower())
    
    article_pattern = r'Điều \d+\.[^#]+'
    articles = re.findall(article_pattern, penalty_text)
    
    relevant_sections = []
    
    if category == "Biển báo cấm":
        relevant_sections.append("""
**Vi phạm biển báo cấm - Điều 6 Nghị định 168/2024/NĐ-CP:**

- **Khoản 1 (400,000 - 600,000 VNĐ)**: Không chấp hành hiệu lệnh, chỉ dẫn của biển báo hiệu, vạch kẻ đường
- **Khoản 5 (4,000,000 - 6,000,000 VNĐ)**: Đi vào khu vực cấm, đường có biển báo hiệu cấm đi vào
- **Khoản 9 (18,000,000 - 20,000,000 VNĐ)**: Đi ngược chiều của đường một chiều, đường có biển "Cấm đi ngược chiều"
""")
    
    if 'P.103' in base_code or 'cấm ô tô' in vietnamese_name.lower():
        relevant_sections.append("""
**Vi phạm cấm ô tô (P.103) - Điều 6 Nghị định 168/2024/NĐ-CP:**

- **Khoản 5i**: Đi vào đường có biển cấm đi vào đối với loại phương tiện đang điều khiển → 4,000,000 - 6,000,000 VNĐ
- **Khoản 10a**: Đi vào đường có biển cấm gây tai nạn giao thông → 20,000,000 - 22,000,000 VNĐ
- Còn bị trừ 2-10 điểm GPLX tùy mức độ
""")
    
    if 'P.104' in base_code or 'cấm mô tô' in vietnamese_name.lower() or 'cấm xe máy' in vietnamese_name.lower():
        relevant_sections.append("""
**Vi phạm cấm xe mô tô/xe máy (P.104) - Điều 7 Nghị định 168/2024/NĐ-CP:**

- **Khoản 6b**: Đi vào khu vực cấm, đường có biển cấm đi vào → 2,000,000 - 3,000,000 VNĐ
- **Khoản 10a**: Đi vào đường cấm gây tai nạn giao thông → 10,000,000 - 14,000,000 VNĐ
- Còn bị trừ 2-10 điểm GPLX tùy mức độ
""")
    
    if 'P.106' in base_code or 'cấm xe tải' in vietnamese_name.lower():
        relevant_sections.append("""
**Vi phạm cấm xe tải (P.106) - Điều 6 Nghị định 168/2024/NĐ-CP:**

- **Khoản 5i**: Đi vào đường có biển cấm xe tải → 4,000,000 - 6,000,000 VNĐ
- **Khoản 10a**: Đi vào đường cấm gây tai nạn → 20,000,000 - 22,000,000 VNĐ
- Xe tải vi phạm còn có thể bị xử phạt theo Điều 24 về quá tải, quá khổ
""")
    
    if 'P.107' in base_code or 'cấm ô tô khách' in vietnamese_name.lower():
        relevant_sections.append("""
**Vi phạm cấm ô tô khách/ô tô tải (P.107) - Điều 6 Nghị định 168/2024/NĐ-CP:**

- **Khoản 5i**: Đi vào đường có biển cấm → 4,000,000 - 6,000,000 VNĐ
- **Khoản 10a**: Vi phạm gây tai nạn → 20,000,000 - 22,000,000 VNĐ
- Còn bị trừ 2 điểm GPLX
""")
    
    if 'P.108' in base_code or 'B.8' in base_code or 'rơ-moóc' in vietnamese_name.lower() or 'sơ-mi' in vietnamese_name.lower():
        relevant_sections.append("""
**Vi phạm cấm xe kéo rơ-moóc (P.108) - Điều 6 Nghị định 168/2024/NĐ-CP:**

- **Khoản 5i**: Đi vào đường có biển cấm xe kéo rơ-moóc → 4,000,000 - 6,000,000 VNĐ
- Còn bị trừ 2 điểm GPLX
""")    
    
    if 'P.111' in base_code or 'hai và ba bánh' in vietnamese_name.lower():
        relevant_sections.append("""
**Vi phạm cấm xe hai/ba bánh (P.111) - Điều 7 Nghị định 168/2024/NĐ-CP:**

- **Khoản 6b**: Đi vào khu vực cấm, đường có biển cấm → 2,000,000 - 3,000,000 VNĐ
- Còn bị trừ 2 điểm GPLX
""")
    
    if 'P.117' in base_code or 'chiều cao' in vietnamese_name.lower():
        relevant_sections.append("""
**Vi phạm giới hạn chiều cao (P.117) - Điều 6 Nghị định 168/2024/NĐ-CP:**

- **Khoản 1a**: Không chấp hành biển báo → 400,000 - 600,000 VNĐ
- **Khoản 5i**: Đi vào đường có biển cấm → 4,000,000 - 6,000,000 VNĐ
- Nếu gây hư hỏng công trình: Bồi thường thiệt hại + Xử lý hình sự nếu nghiêm trọng
""")
    
    if 'P.127' in base_code or 'tốc độ' in vietnamese_name.lower():
        relevant_sections.append("""
**Vi phạm tốc độ - Điều 6 Nghị định 168/2024/NĐ-CP:**

Xe ô tô:
- Khoản 3a: Vượt 5-10 km/h → 800,000 - 1,000,000 VNĐ
- Khoản 5đ: Vượt 10-20 km/h → 4,000,000 - 6,000,000 VNĐ  
- Khoản 6a: Vượt 20-35 km/h → 6,000,000 - 8,000,000 VNĐ
- Khoản 7a: Vượt trên 35 km/h → 12,000,000 - 14,000,000 VNĐ + Trừ 6-10 điểm GPLX

Xe mô tô (Điều 7):
- Khoản 2b: Vượt 5-10 km/h → 400,000 - 600,000 VNĐ
- Khoản 4a: Vượt 10-20 km/h → 800,000 - 1,000,000 VNĐ
- Khoản 8a: Vượt trên 20 km/h → 6,000,000 - 8,000,000 VNĐ
""")
    
    if 'P.129' in base_code or 'kiểm tra' in vietnamese_name.lower():
        relevant_sections.append("""
**Vi phạm trạm kiểm tra (P.129) - Nghị định 168/2024/NĐ-CP:**

- Không dừng xe tại trạm kiểm tra khi có yêu cầu
- **Điều 6 Khoản 9c**: Không chấp hành hiệu lệnh của người kiểm soát giao thông → 18,000,000 - 20,000,000 VNĐ
- Còn bị trừ 4 điểm GPLX
""")
    
    if 'rẽ' in vietnamese_name.lower() or 'P.123' in base_code or 'P.137' in base_code:
        relevant_sections.append("""
**Vi phạm rẽ trái/phải - Điều 6 Nghị định 168/2024/NĐ-CP:**

- **Khoản 4k**: Điều khiển xe rẽ trái tại nơi có biển cấm rẽ trái, rẽ phải tại nơi có biển cấm rẽ phải → 2,000,000 - 3,000,000 VNĐ
- Còn bị trừ 2 điểm GPLX
""")
    
    if 'quay đầu' in vietnamese_name.lower() or 'P.124' in base_code:
        relevant_sections.append("""
**Vi phạm quay đầu xe - Điều 6 Nghị định 168/2024/NĐ-CP:**

- **Khoản 4k**: Quay đầu xe tại nơi có biển cấm quay đầu → 2,000,000 - 3,000,000 VNĐ
- **Khoản 4i**: Quay đầu xe trên cầu, đầu cầu, trong hầm, đường hẹp, đường dốc, đường cong → 2,000,000 - 3,000,000 VNĐ
- Còn bị trừ 2 điểm GPLX
""")
    
    if 'dừng' in vietnamese_name.lower() or 'đỗ' in vietnamese_name.lower() or 'P.130' in base_code or 'P.131' in base_code:
        relevant_sections.append("""
**Vi phạm dừng/đỗ xe - Điều 6 Nghị định 168/2024/NĐ-CP:**

- **Khoản 2đ**: Dừng xe nơi có biển "Cấm dừng xe và đỗ xe" → 600,000 - 800,000 VNĐ
- **Khoản 3e**: Đỗ xe nơi có biển "Cấm đỗ xe" hoặc "Cấm dừng xe và đỗ xe" → 800,000 - 1,000,000 VNĐ
- **Khoản 5k**: Dừng xe, đỗ xe gây ùn tắc giao thông → 4,000,000 - 6,000,000 VNĐ
""")
    
    if 'ngược chiều' in vietnamese_name.lower() or 'P.102' in base_code:
        relevant_sections.append("""
**Vi phạm đi ngược chiều - Điều 6 Nghị định 168/2024/NĐ-CP:**

Xe ô tô:
- **Khoản 9d**: Đi ngược chiều của đường một chiều, đường có biển "Cấm đi ngược chiều" → 18,000,000 - 20,000,000 VNĐ + Trừ 4 điểm GPLX

Xe mô tô (Điều 7):
- **Khoản 7a**: Đi ngược chiều của đường một chiều, đường có biển "Cấm đi ngược chiều" → 4,000,000 - 6,000,000 VNĐ + Trừ 2 điểm GPLX
""")
    
    if category == "Biển báo nguy hiểm":
        relevant_sections.append("""
**Biển báo nguy hiểm (W.) - Lưu ý:**

Biển báo nguy hiểm không quy định mức phạt trực tiếp, nhưng:
- Không giảm tốc độ khi gặp biển cảnh báo có thể bị phạt theo Điều 6 Khoản 3o
- Gây tai nạn tại vị trí có biển cảnh báo: Tăng nặng hình phạt
- Lái xe cần tuân thủ để đảm bảo an toàn
""")
    
    if category == "Biển báo hiệu lệnh":
        relevant_sections.append("""
**Biển báo hiệu lệnh (R.) - Điều 6 Nghị định 168/2024/NĐ-CP:**

- **Khoản 1a**: Không chấp hành biển báo hiệu lệnh → 400,000 - 600,000 VNĐ
- Biển hiệu lệnh yêu cầu phải tuân thủ hướng đi, làn đường quy định
""")
    
    if relevant_sections:
        return "\n".join(relevant_sections)
    
    return None


def get_sign_info(class_id: int) -> dict:
    LABEL_CHAR, LABEL_TEXT = load_classes()
    
    if class_id not in LABEL_CHAR:
        return {"error": f"Class ID {class_id} not found"}
    
    sign_code = LABEL_CHAR[class_id]
    vietnamese_name = LABEL_TEXT[class_id]
    base_code = extract_sign_code(sign_code)
    category = get_sign_category(sign_code)
    
    regulations_text = load_sign_regulations()
    penalty_text = load_penalty_law()
    
    regulation_info = search_sign_in_regulations(sign_code, regulations_text)
    penalty_info = get_related_penalties(sign_code, vietnamese_name, penalty_text)
    
    result = {
        "class_id": class_id,
        "sign_code": sign_code,
        "base_code": base_code,
        "vietnamese_name": vietnamese_name,
        "category": category,
        "regulation_info": regulation_info,
        "penalty_info": penalty_info
    }
    
    return result


def format_sign_info(info: dict) -> str:
    lines = [
        "=" * 60,
        f"THÔNG TIN BIỂN BÁO GIAO THÔNG",
        "=" * 60,
        f"",
        f" **Mã biển**: {info['sign_code']}",
        f" **Tên gọi**: {info['vietnamese_name']}",
        f" **Phân loại**: {info['category']}",
        f"",
    ]
    
    if info.get('regulation_info'):
        lines.extend([
            "-" * 60,
            " **QUY ĐỊNH CHI TIẾT (QCVN 41:2019/BGTVT)**",
            "-" * 60,
            info['regulation_info'],
            ""
        ])
    
    if info.get('penalty_info'):
        lines.extend([
            "-" * 60,
            " **MỨC XỬ PHẠT (Nghị định 168/2024/NĐ-CP)**",
            "-" * 60,
            info['penalty_info'],
            ""
        ])
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def lookup_sign(class_id: int) -> str:
    info = get_sign_info(class_id)
    
    if "error" in info:
        return f"Lỗi: {info['error']}"
    
    return format_sign_info(info)


def lookup_by_code(sign_code: str) -> str:
    """Look up sign by its code (e.g., 'P.102')"""
    LABEL_CHAR, _ = load_classes()
    
    for class_id, code in LABEL_CHAR.items():
        if extract_sign_code(code) == extract_sign_code(sign_code):
            return lookup_sign(class_id)
    
    return f"Không tìm thấy biển báo với mã: {sign_code}"


def lookup_by_name(name_keyword: str) -> str:
    """Look up sign by name keyword"""
    _, LABEL_TEXT = load_classes()
    
    results = []
    for class_id, name in LABEL_TEXT.items():
        if name_keyword.lower() in name.lower():
            results.append(f"ID {class_id}: {name}")
    
    if results:
        return "Các biển báo tìm thấy:\n" + "\n".join(results)
    
    return f"Không tìm thấy biển báo với từ khóa: {name_keyword}"


if __name__ == "__main__":
    print(lookup_sign(17))  

