import os

def rename_directories(path='.'):
    for root, dirs, files in os.walk(path, topdown=False):
        for directory in dirs:
            old_dir_path = os.path.join(root, directory)
            # Sostituisce '?' e ':' con '_'
            new_dir_name = directory.replace('?', '_').replace(':', '_')
            new_dir_path = os.path.join(root, new_dir_name)

            # Se il nome è cambiato, rinomina la directory
            if old_dir_path != new_dir_path:
                print(f'Renaming: {old_dir_path} -> {new_dir_path}')
                os.rename(old_dir_path, new_dir_path)

if __name__ == '__main__':
    # Avvia la funzione a partire dalla directory corrente
    rename_directories('IngegneriaScienzeInformatiche')
    rename_directories('SviluppoCooperazioneInternazionale')
