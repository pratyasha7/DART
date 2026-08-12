import os
import tarfile
import urllib.request

# 1. FIXED: Appended '?download=1' to bypass the Zenodo HTML landing page and stream the raw binary
url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1"
output_archive = "urbansound8k.tar.gz"

# 2. FIXED: Updated size text to accurately reflect the 6.0 GB compressed archive size
print("Downloading UrbanSound8K (approx. 6.0 GB)... This will take a while depending on your connection.")

# A basic progress reporter with boundary checks
def progress_reporter(block_num, block_size, total_size):
    downloaded = block_num * block_size
    dl_gb = downloaded / (1024**3)
    
    # 3. FIXED: Added logic to prevent ZeroDivisionError or negative size values
    if total_size is not None and total_size > 0:
        percent = (downloaded / total_size) * 100
        total_gb = total_size / (1024**3)
        print(f"\rDownloading: {percent:.2f}% ({dl_gb:.2f} GB of {total_gb:.2f} GB)", end="")
    else:
        # Fallback print format if the server does not report 'Content-Length' initially
        print(f"\rDownloading: {dl_gb:.2f} GB (Total size unknown)", end="")

try:
    # Optional: Set a standard User-Agent header so Zenodo doesn't reject the script as a generic bot
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
    urllib.request.install_opener(opener)
    
    # Run the direct download
    urllib.request.urlretrieve(url, output_archive, reporthook=progress_reporter)
    print("\nDownload complete! Beginning extraction (this may take a few minutes)...")
    
    # Extract the true .tar.gz binary
    with tarfile.open(output_archive, "r:gz") as tar:
        tar.extractall()
    print("Extraction complete!")
    
    # Clean up to save 6.0 GB of disk space
    if os.path.exists(output_archive):
        os.remove(output_archive)
        print("Temporary compressed archive deleted successfully.")
    
except Exception as e:
    print(f"\nAn error occurred: {e}")