import os
import time
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader


print("Loading documents from ./banking ...")
loader = DirectoryLoader(
    "./banking",
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()
print(f"Loaded {len(documents)} documents.")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=75
)
splits = text_splitter.split_documents(documents)
print(f"Total chunks after splitting: {len(splits)}")


vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./banking_rag_db"
)


retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


llm = ChatOllama(
    model="llama3",
    temperature=0,
    timeout=300
)


system_prompt = (
    "You are a helpful banking assistant. Use the provided context "
    "to answer customer queries accurately. Do not provide harmful or misleading information. "
    "If you don't know the answer, say that you don't know.\n\n"
    "Context: {context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt_template
    | llm
    | StrOutputParser()
)

chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)


input_file = "benign_prompt.txt"
output_file = "benign_lc_result"

print(f"Starting batch execution...")

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "a", encoding="utf-8") as outfile:
    for line in infile:
        question = line.strip()
        if not question:
            continue

        print(f"Processing: {question}")

        try:
            start_time = time.time()
            result = chain.invoke(question)
            latency = time.time() - start_time

            outfile.write(f"Prompt: {question}\n")
            outfile.write(f"Latency: {latency:.2f} seconds\n")
            outfile.write("Retrieved Chunks:\n")

            for i, doc in enumerate(result["context"]):
                
                source = doc.metadata.get("source", "unknown")
                outfile.write(f"--- Chunk {i+1} | Source: {source} ---\n")
                outfile.write(f"{doc.page_content}\n")

            outfile.write(f"\nResponse:\n{result['answer']}\n")
            outfile.write("=" * 60 + "\n")
            outfile.flush()
            os.fsync(outfile.fileno())

        except Exception as e:
            error_msg = f"Error processing prompt '{question}': {str(e)}"
            print(error_msg)
            outfile.write(f"Prompt: {question}\n")
            outfile.write(f"Error: {error_msg}\n")
            outfile.write("=" * 60 + "\n")
            outfile.flush()
            os.fsync(outfile.fileno())

print("Execution complete.")