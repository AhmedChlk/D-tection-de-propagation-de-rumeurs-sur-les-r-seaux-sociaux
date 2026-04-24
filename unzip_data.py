import zipfile
import os

def extract_data():
    zip_path = "data/Rumor-Detection-Dataset.zip"
    extract_dir = "data/kaggle/Rumor-Detection-Dataset/"

    if not os.path.exists(zip_path):
        print(f"[ERREUR] Le fichier {zip_path} est introuvable.")
        return

    print(f"Extraction de {zip_path} vers {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Extraction terminée avec succès !")

if __name__ == "__main__":
    extract_data()