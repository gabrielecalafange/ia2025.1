from scholarly import scholarly
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time


# Busca por um tópico
search_query = scholarly.search_pubs('Diagnostic of viral pneumonia')

# 3. Headers para fingir ser um navegador
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

count = 0
for i in search_query:  

    # Limita a 50 downloads
    if count >= 50:
        break

    first_result = next(search_query)

    title = first_result['bib']['title']
    url = first_result.get('eprint_url') or first_result.get('pub_url')

    # 2. Limpar o título para ser um nome de arquivo válido
    filename = re.sub(r'[\\/*?:"<>|]', "", title) + ".pdf"

    print(f"Tentando baixar: {title}")
    print(f"URL: {url}")
    
    url = i.get('eprint_url') or i.get('pub_url')
    filename = re.sub(r'[\\/*?:"<>|]', "", i['bib']['title']) + ".pdf"
    try:
        response = requests.get(url, headers=headers, timeout=15)

        response = requests.get(url, headers=headers)
        response.raise_for_status() # Verifica se houve erro no download

        
        with open(filename, "wb") as f:
            f.write(response.content)
            print("Download concluído com sucesso!")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        pdf_link = None
        for link in soup.find_all('a', href=True):
            if '.pdf' in link['href'].lower() or 'download' in link.get('title', '').lower():
                pdf_link = urljoin(url, link['href'])
                break

            if pdf_link:
                response = requests.get(pdf_link, headers=headers, timeout=15)
                with open(filename, "wb") as f:
                    f.write(response.content)
                    print(f"Sucesso! Arquivo salvo como: {filename}")
            else:
                print("Não foi possível encontrar um link PDF na página.")
                break
    
    except Exception as e:
        print(f"Falha ao baixar o arquivo: {e}")