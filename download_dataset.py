import os
import tarfile
import urllib.request

url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"
output_archive = "urbansound8k.tar.gz"


print("Downloading UrbanSound8K (approx. 6.0 GB)... This will take a while depending on your connection.")

def progress_reporter(block_num, block_size, total_size):
    downloaded = block_num * block_size
    dl_gb = downloaded / (1024**3)

    if total_size is not None and total_size > 0:
        percent = (downloaded / total_size) * 100
        total_gb = total_size / (1024**3)
        print(f"\rDownloading: {percent:.2f}% ({dl_gb:.2f} GB of {total_gb:.2f} GB)", end="")
    else:
       
        print(f"\rDownloading: {dl_gb:.2f} GB (Total size unknown)", end="")

try:
    
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)

    urllib.request.urlretrieve(url, output_archive, reporthook=progress_reporter)
    print("\nDownload complete! Beginning extraction (this may take a few minutes)...")

    with tarfile.open(output_archive, "r:gz") as tar:
        tar.extractall()
    print("Extraction complete!")

    if os.path.exists(output_archive):
        os.remove(output_archive)
        print("Temporary compressed archive deleted successfully.")
    
except Exception as e:
    print(f"\nAn error occurred: {e}")