import requests
import re
import os
import time

TERMO_BUSCA = 'Diagnostic of viral pneumonia'
LIMITE = 20
PASTA_DESTINO = "papers_pneumonia"

if not os.path.exists(PASTA_DESTINO):
    os.makedirs(PASTA_DESTINO)

def buscar_crossref(query, limit=50):
    # API da Crossref não precisa de chave, mas identificar seu email ajuda na prioridade
    url = f"https://api.crossref.org/works?query={query}&rows={limit}"
    headers = {"User-Agent": "ramoni.negereiros (mailto:ramoni.reus.barros.negreiros@ccc.ufcg.edu.br)"}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['message']['items']
    return []

def baixar_artigo(url_lista, nome_arquivo):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url in url_lista:
        try:
            res = requests.get(url, headers=headers, timeout=15)
            if 'application/pdf' in res.headers.get('Content-Type', '').lower():
                with open(nome_arquivo, "wb") as f:
                    f.write(res.content)
                return True
        except:
            continue
    return False

# --- Execução ---
artigos = buscar_crossref(TERMO_BUSCA, LIMITE)
print(f"Encontrados {len(artigos)} registros na Crossref.")

baixados = 0
for item in artigos:
    if baixados >= LIMITE: break
    
    titulo = item.get('title', ['Sem Titulo'])[0]
    # A Crossref fornece o link da editora (resource) e links diretos (link)
    links = []
    if 'link' in item:
        links = [l['URL'] for l in item['link'] if l['content-type'] == 'application/pdf']
    if 'URL' in item:
        links.append(item['URL'])

    clean_title = re.sub(r'[\\/*?:"<>|]', "", titulo)[:80]
    caminho = os.path.join(PASTA_DESTINO, f"{clean_title}.pdf")

    print(f"[{baixados+1}] Tentando: {titulo[:50]}...")
    
    if baixar_artigo(links, caminho):
        print("   ✅ PDF salvo!")
        baixados += 1
    else:
        print("   ❌ PDF não disponível ou pago.")
    
    time.sleep(1) # Delay mínimo, Crossref é generosa

print(f"\nConcluído! Total de PDFs baixados: {baixados}")