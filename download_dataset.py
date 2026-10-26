"""Download and extract the anime faces dataset (needs Kaggle credentials, see the README)."""
import kaggle

if __name__ == "__main__":
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files("splcher/animefacedataset", path="./", unzip=True)
