import os

from tqdm import tqdm


def sync_directories(parent_dir, guide_dir):
    # Obtiene la lista de archivos en el directorio guía
    guide_files = set(os.listdir(guide_dir))

    # Itera a través de todas las subcarpetas en el directorio padre
    for root, dirs, files in tqdm(os.walk(parent_dir)):
        for file in files:
            # Si el archivo no está en el directorio guía, lo eliminamos
            if file not in guide_files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Eliminado: {file_path}")

def sync_revisitis(guide_photos_dir, revisits_dir):

    guide = []
    for file in tqdm(os.listdir(guide_photos_dir)):
        pic_name = file[:-4]
        guide.append(pic_name)

    for revisit in tqdm(os.listdir(revisits_dir)):
        if revisit[:-6] not in guide:
            file_path = os.path.join(revisits_dir, revisit)
            os.remove(file_path)





if __name__ == '__main__':
    sync_directories("E:\\TFG\\dataset_tfg", "E:\\TFG\\dataset_tfg\\256")