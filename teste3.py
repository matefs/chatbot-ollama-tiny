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
from datetime import datetime

# Configuração da página
st.set_page_config(page_title="Assistente de Jornada - E-ponto", layout="wide")

# Inicialização dos estados
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []

# Credenciais
USERNAME = "admin"
PASSWORD = "1234"

# Função para limpar o chat
def clear_chat():
    st.session_state.messages = []

# Tela de login
if not st.session_state["authenticated"]:
    st.title("Login - Assistente de Jornada")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        
        if st.button("Entrar", use_container_width=True):
            if username == USERNAME and password == PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Usuário ou senha incorretos!")
    st.stop()

# Configuração do prompt e do modelo
rag_template = """
Você é um atendente de uma empresa. Seu trabalho é conversar com os clientes, consultando a base de conhecimentos da empresa, e dar uma resposta simples e precisa para ele, baseada na base de dados da empresa fornecida como contexto. Considere o "resolution_date" como a data de atualização da informação.
Contexto: {context}
Pergunta do cliente: {question}
"""

# Carregar o modelo local (Ollama)
ollama_server_url = "http://localhost:11434"
model_local = ChatOllama(base_url=ollama_server_url, model="llama3")

@st.cache_data
def load_json_data():
    documents = []
    json_directory = Path(".")
    
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

# Sidebar com menu
with st.sidebar:
    st.title("Menu")
    menu_option = st.radio("", ["Chat", "Histórico de Prompts"])
    st.divider()
    if st.button("Novo Chat", use_container_width=True):
        clear_chat()
    if st.button("Sair", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

if menu_option == "Chat":
    st.title("Assistente de Jornada - E-ponto")

    # Container para as mensagens
    chat_container = st.container()
    
    # Container para o input
    with st.container():
        # Criar uma linha horizontal para separar
        st.divider()
        
        # Input da pergunta do cliente
        col1, col2 = st.columns([8,1])
        with col1:
            question = st.text_input("", placeholder="Digite sua mensagem...", label_visibility="collapsed")
        with col2:
            send_button = st.button("Enviar")

    # Exibir mensagens anteriores
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Processar nova mensagem
    if question and send_button:
        # Adicionar mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Gerar e adicionar resposta
        with st.spinner("Processando..."):
            response = chatbot_response(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Adicionar ao histórico
            st.session_state["history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pergunta": question,
                "resposta": response
            })
        
        st.rerun()

elif menu_option == "Histórico de Prompts":
    st.title("Histórico de Conversas")
    
    if not st.session_state["history"]:
        st.info("Nenhuma conversa registrada ainda.")
    else:
        for item in reversed(st.session_state["history"]):
            with st.expander(f"Conversa em {item['timestamp']}"):
                st.write("**Pergunta:**")
                st.write(item["pergunta"])
                st.write("**Resposta:**")
                st.write(item["resposta"])
        
        if st.button("Limpar Histórico", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()
