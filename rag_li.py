import os
import time

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb


print("Loading documents from ./banking ...")
documents = SimpleDirectoryReader(
    input_dir="./banking",
    required_exts=[".txt"],
    recursive=True
).load_data()
print(f"Loaded {len(documents)} documents.")


embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


llm = Ollama(
    model="llama3",
    temperature=0,
    request_timeout=300
)


Settings.llm = llm
Settings.embed_model = embed_model


splitter = SentenceSplitter(
    chunk_size=350,
    chunk_overlap=75
)
nodes = splitter.get_nodes_from_documents(documents)
print(f"Total chunks after splitting: {len(nodes)}")


chroma_client = chromadb.PersistentClient(path="./banking_rag_db_li")
chroma_collection = chroma_client.get_or_create_collection("banking_llamaindex")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
)


query_engine = index.as_query_engine(
    similarity_top_k=5,
   
    system_prompt=(
        "You are a helpful banking assistant. Use the provided context "
        "to answer customer queries accurately. Do not provide harmful or misleading information. "
        "If you don't know the answer, say that you don't know."
    )
)


input_file = "benign_prompt.txt"
output_file = "bening_li_result"  

print("Starting batch execution...")

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "a", encoding="utf-8") as outfile:

    for line in infile:
        question = line.strip()
        if not question:
            continue

        print(f"Processing: {question}")

        try:
            start_time = time.time()
            result = query_engine.query(question)
            latency = time.time() - start_time

            outfile.write(f"Prompt: {question}\n")
            outfile.write(f"Latency: {latency:.2f} seconds\n")
            outfile.write("Retrieved Chunks:\n")

            
            for i, node in enumerate(result.source_nodes):
                source = node.metadata.get("file_path", "unknown")
                score = node.score if node.score is not None else 0.0
                outfile.write(f"--- Chunk {i+1} | Source: {source} | Score: {score:.4f} ---\n")
                outfile.write(f"{node.text}\n")

            outfile.write(f"\nResponse:\n{str(result)}\n")
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