import gdown

url = "https://drive.google.com/drive/folders/1xG5iyvHlf4OgUwZUr_JYf7_-fFFjOyZ5"
gdown.download_folder(url, output="datasets", quiet=False)
