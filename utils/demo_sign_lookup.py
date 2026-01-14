import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sign_info_parser import lookup_sign, lookup_by_code, lookup_by_name, get_sign_info
from utils.label_const import LABEL_TEXT, LABEL_CHAR


def demo_all_signs():
    print("=" * 70)
    print("DANH SÁCH 52 LOẠI BIỂN BÁO GIAO THÔNG ĐƯỢC PHÁT HIỆN")
    print("=" * 70)
    print(f"\n{'ID':<4} {'Mã biển':<18} {'Tên gọi tiếng Việt'}")
    print("-" * 70)
    
    for i in range(1, 53):
        code = LABEL_CHAR.get(i, "N/A")
        name = LABEL_TEXT.get(i, "N/A")
        print(f"{i:<4} {code:<18} {name}")
    
    print("-" * 70)


def demo_specific_signs():
    """Demo looking up specific signs"""
    # Common prohibition signs
    test_signs = [
        3,   # P.102 - Cấm đi ngược chiều
        8,   # P.123a - Cấm rẽ trái
        11,  # P.130 - Cấm dừng và đỗ xe
        18,  # P.124a - Cấm quay đầu
        39,  # P.127*50 - Giới hạn tốc độ 50km/h
    ]
    
    print("\n\n" + "=" * 70)
    print("DEMO: TRA CỨU THÔNG TIN CHI TIẾT CÁC BIỂN BÁO PHỔ BIẾN")
    print("=" * 70)
    
    for sign_id in test_signs:
        print(lookup_sign(sign_id))
        print("\n")


def demo_search():
    """Demo searching by name"""
    print("\n" + "=" * 70)
    print("DEMO: TÌM KIẾM BIỂN BÁO THEO TÊN")
    print("=" * 70)
    
    keywords = ["cấm", "tốc độ", "rẽ", "đỗ"]
    
    for kw in keywords:
        print(f"\n🔍 Tìm kiếm: '{kw}'")
        print("-" * 40)
        result = lookup_by_name(kw)
        print(result)


def interactive_mode():
    """Interactive lookup mode"""
    print("\n" + "=" * 70)
    print("CHẾ ĐỘ TRA CỨU TƯƠNG TÁC")
    print("=" * 70)
    print("Nhập ID biển báo (1-52) hoặc mã biển (VD: P.102)")
    print("Nhập 'list' để xem danh sách, 'quit' để thoát\n")
    
    while True:
        user_input = input(">>> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Tạm biệt!")
            break
        elif user_input.lower() == 'list':
            demo_all_signs()
        elif user_input.isdigit():
            class_id = int(user_input)
            if 1 <= class_id <= 52:
                print(lookup_sign(class_id))
            else:
                print(" ID phải từ 1 đến 52")
        elif user_input.startswith(('P.', 'W.', 'R.', 'I.', 'S.')):
            print(lookup_by_code(user_input))
        else:
            print(lookup_by_name(user_input))
        
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo sign information lookup")
    parser.add_argument("--list", action="store_true", help="List all signs")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--search", type=str, help="Search by keyword")
    parser.add_argument("--id", type=int, help="Lookup by class ID")
    parser.add_argument("--code", type=str, help="Lookup by sign code")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.list:
        demo_all_signs()
    elif args.demo:
        demo_all_signs()
        demo_specific_signs()
        demo_search()
    elif args.search:
        print(lookup_by_name(args.search))
    elif args.id:
        print(lookup_sign(args.id))
    elif args.code:
        print(lookup_by_code(args.code))
    elif args.interactive:
        interactive_mode()
    else:
        demo_all_signs()
        demo_specific_signs()
