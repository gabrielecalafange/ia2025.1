import os
from flask import Flask, render_template, request, jsonify
from se import diagnosticar # Importa a função do seu sistema especialista

# Descobre o caminho absoluto de onde o app.py está rodando
diretorio_atual = os.path.dirname(os.path.abspath(__file__))
# Aponta explicitamente para a pasta 'templates' que deve estar no mesmo diretório
pasta_templates = os.path.join(diretorio_atual, "templates")

# Inicializa o Flask já ensinando onde estão os templates
app = Flask(__name__, template_folder=pasta_templates)

# Rota da página inicial
@app.route("/")
def home():
    return render_template("index.html")

# Rota da API que recebe os dados do form e retorna o diagnóstico
@app.route("/diagnostico", methods=["POST"])
def diagnostico_route():
    try:
        # Pega os dados JSON enviados pelo JavaScript (fetch)
        dados_paciente = request.json
        
        # Chama a função diagnosticar do se.py passando o dicionário
        resultado = diagnosticar(dados_paciente)
        
        # Devolve o resultado em formato JSON para o front-end
        return jsonify({"diagnostico": resultado}), 200

    except Exception as e:
        # Caso falte algum dado ou dê erro no processamento
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)