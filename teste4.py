import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama
from llama_index.llms.ollama import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader

# Configuração do prompt e do modelo
rag_template = """
Você é um atendente de uma empresa. Seu trabalho é conversar com os clientes, consultando a base de conhecimentos da empresa, e dar uma resposta simples e precisa para ele, baseada na base de dados da empresa fornecida como contexto.
Contexto: {context}
Pergunta do cliente: {question}
"""

# Carregar o modelo local (Ollama)
ollama_server_url = "http://localhost:11434"  # IP configurado para localhost
model_local = ChatOllama(base_url=ollama_server_url, model="llama3")

# Função para carregar a base de conhecimento
@st.cache_data
def load_csv_data():
    loader = CSVLoader(file_path="./knowledge_base.csv")  # Altere para o seu arquivo CSV
    embeddings = OllamaEmbeddings(base_url=ollama_server_url, model="tinyllama:latest")
    documents = loader.load()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# Carregar a base de dados
retriever = load_csv_data()

# Configuração do prompt para o chatbot
prompt = ChatPromptTemplate.from_template(rag_template)

# Função ajustada para incluir o contexto da base de conhecimento
def chatbot_response(question):
    # Recuperar o contexto relevante da base de dados
    relevant_docs = retriever.get_relevant_documents(question)
    context = " ".join([doc.page_content for doc in relevant_docs])
    
    # Criar o prompt com o contexto e a pergunta
    final_prompt = prompt.format(context=context, question=question)
    
    # Ajustar parâmetros do modelo
    model_kwargs = {
        "temperature": 0.7,
        "max_output_tokens": 500  # Reduzir tokens de saída se necessário
    }
    
    # Obter a resposta do modelo
    response = model_local.predict(final_prompt, **model_kwargs)
    return response

# Interface no Streamlit
st.title("Oráculo - Asimov Academy")

# Input da pergunta do cliente
question = st.text_input("Digite sua pergunta:")

# Geração da resposta
if question:
    resposta = chatbot_response(question)
    st.write(resposta)

