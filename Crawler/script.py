import os

def convert_html_to_txt(root_folder):
    # Camminare attraverso tutte le cartelle e sottocartelle
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.html'):
                html_file_path = os.path.join(foldername, filename)
                
                # Genera il nuovo nome del file con estensione .txt
                txt_file_path = os.path.join(foldername, filename.replace('.html', '.txt'))
                
                # Legge il contenuto del file .html
                with open(html_file_path, 'r', encoding='utf-8') as html_file:
                    content = html_file.read()
                
                # Scrive lo stesso contenuto in un file .txt
                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(content)
                
                # Rimuove il file HTML originale
                os.remove(html_file_path)

                print(f'Convertito: {html_file_path} -> {txt_file_path}')
