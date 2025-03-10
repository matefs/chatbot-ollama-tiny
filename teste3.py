import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from llama_index.llms.ollama import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
import os
from pathlib import Path

# Configuração da página
st.set_page_config(page_title="Assistente de Jornada - E-ponto", layout="wide")

# Inicialização do estado de autenticação
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Credenciais
USERNAME = "admin"
PASSWORD = "1234"

# Tela de login
if not st.session_state["authenticated"]:
    st.title("Login - Assistente de Jornada")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    
    if st.button("Entrar"):
        if username == USERNAME and password == PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos!")
    st.stop()

# O resto do código só será executado se estiver autenticado
# Configuração do prompt e do modelo
rag_template = """
Você é um atendente de uma empresa. Seu trabalho é conversar com os clientes, consultando a base de conhecimentos da empresa, e dar uma resposta simples e precisa para ele, baseada na base de dados da empresa fornecida como contexto. Considere o "resolution_date" como a data de atualização da informação.
Contexto: {context}
Pergunta do cliente: {question}
"""

# Carregar o modelo local (Ollama)
ollama_server_url = "http://localhost:11434"
model_local = ChatOllama(base_url=ollama_server_url, model="llama3")

# Função para carregar a base de conhecimento
@st.cache_data
def load_json_data():
    documents = []
    json_directory = Path(".")  # diretório atual
    
    for json_file in json_directory.glob("*.json"):
        loader = JSONLoader(
            file_path=str(json_file),
            jq_schema=".[]",
            text_content=False
        )
        documents.extend(loader.load())
    
    embeddings = OllamaEmbeddings(base_url=ollama_server_url, model="tinyllama:latest")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# Carregar a base de dados
retriever = load_json_data()

# Configuração do prompt para o chatbot
prompt = ChatPromptTemplate.from_template(rag_template)

# Example adjustment in your chatbot_response function
def chatbot_response(question):
    relevant_docs = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in relevant_docs])
    
    final_prompt = prompt.format(context=context, question=question)
    
    model_kwargs = {
        "temperature": 0.7,
        "max_output_tokens": 500
    }
    response = model_local.predict(final_prompt, **model_kwargs)
    return response

# Interface principal do chat
st.title("Assistente de Jornada - E-ponto")

# Botão de logout na sidebar
if st.sidebar.button("Sair"):
    st.session_state["authenticated"] = False
    st.rerun()

# Input da pergunta do cliente
question = st.text_input("Digite sua pergunta:")

# Geração da resposta
if question:
    with st.spinner("Processando sua pergunta..."):
        resposta = chatbot_response(question)
    st.write(resposta)
