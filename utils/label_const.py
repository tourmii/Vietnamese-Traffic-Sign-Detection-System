class_file_text =  open("/home/tourmii/Documents/Projects/Traffic_Sign/datasets/classes_vie.txt", "r")
class_file_char = open("/home/tourmii/Documents/Projects/Traffic_Sign/datasets/classes.txt", "r")

labels_text = class_file_text.read().splitlines()
LABEL_TEXT = {i+1: label for i, label in enumerate(labels_text)}

labels_char = class_file_char.read().splitlines()
LABEL_CHAR = {i+1: label for i, label in enumerate(labels_char)}



