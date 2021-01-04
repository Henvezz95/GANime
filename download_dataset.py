import kaggle

kaggle.api.authenticate()
kaggle.api.dataset_download_files('splcher/animefacedataset', path='./', unzip=True)