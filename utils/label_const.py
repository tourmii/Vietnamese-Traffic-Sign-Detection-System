import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

classes_char_file = os.path.join(PROJECT_ROOT, "utils/classes.txt")
classes_vie_file = os.path.join(PROJECT_ROOT, "utils/classes_vie.txt")

with open(classes_char_file, 'r', encoding='utf-8') as f:
    labels_char = [line.strip() for line in f.readlines()]

with open(classes_vie_file, 'r', encoding='utf-8') as f:
    labels_text = [line.strip() for line in f.readlines()]

LABEL_CHAR = {i+1: label for i, label in enumerate(labels_char)}
LABEL_TEXT = {i+1: label for i, label in enumerate(labels_text)}

NUM_CLASSES = len(labels_char)


def get_label_text(class_id: int) -> str:
    """Get Vietnamese name for a class ID"""
    return LABEL_TEXT.get(class_id, "Unknown")


def get_label_char(class_id: int) -> str:
    """Get sign code for a class ID"""
    return LABEL_CHAR.get(class_id, "Unknown")

if __name__ == "__main__":
    print(f"Number of classes: {NUM_CLASSES}")
    print("\nLABEL_TEXT (Vietnamese names):")
    for k, v in LABEL_TEXT.items():
        print(f"  {k}: {v}")
    print("\nLABEL_CHAR (Sign codes):")
    for k, v in LABEL_CHAR.items():
        print(f"  {k}: {v}")