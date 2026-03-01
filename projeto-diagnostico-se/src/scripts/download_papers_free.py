import requests
import time
import os
import re
import xml.etree.ElementTree as ET

TERMO_BUSCA = 'viral pneumonia diagnosis' # Termos em inglês funcionam melhor
LIMITE = 10
PASTA_DESTINO = "papers_ia_saude"

if not os.path.exists(PASTA_DESTINO):
    os.makedirs(PASTA_DESTINO)

def buscar_e_baixar_arxiv(query, max_results=50):
    # API do arXiv usa sintaxe específica
    base_url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}'
    
    response = requests.get(base_url)
    if response.status_code != 200:
        print("Erro ao acessar a API do arXiv")
        return

    # O arXiv retorna XML
    root = ET.fromstring(response.content)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    
    baixados = 0
    for entry in root.findall('atom:entry', ns):
        if baixados >= max_results: break
        
        titulo = entry.find('atom:title', ns).text.strip().replace('\n', '')
        # Busca o link do PDF dentro das tags de link
        pdf_url = ""
        for link in entry.findall('atom:link', ns):
            if link.get('title') == 'pdf':
                pdf_url = link.get('href')
                break
        
        if not pdf_url: continue

        # Limpeza de nome de arquivo
        clean_title = re.sub(r'[\\/*?:"<>|]', "", titulo)[:80]
        caminho_arquivo = os.path.join(PASTA_DESTINO, f"{clean_title}.pdf")

        print(f"[{baixados+1}] Baixando: {titulo[:50]}...")
        
        try:
            # O arXiv pede um pequeno delay entre downloads para não sobrecarregar
            r = requests.get(pdf_url, timeout=30)
            if r.status_code == 200:
                with open(caminho_arquivo, "wb") as f:
                    f.write(r.content)
                print("   ✅ Download concluído!")
                baixados += 1
                time.sleep(2) # Delay ético
            else:
                print(f"   ❌ Falha (Status {r.status_code})")
        except Exception as e:
            print(f"   ⚠️ Erro: {e}")

buscar_e_baixar_arxiv(TERMO_BUSCA, LIMITE)