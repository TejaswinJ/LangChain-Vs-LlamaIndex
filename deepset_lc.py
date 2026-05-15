import os
import csv
import time
import warnings
from datasets import load_dataset


from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


warnings.filterwarnings("ignore", category=UserWarning)


LIMIT = 100
FILENAME = "attack_results_langchain.csv"
MODEL_NAME = "llama3.1:8b"

print(f"Initializing LangChain with {MODEL_NAME}...")

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.0,
    timeout=300.0, 
)


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "{attack_prompt}")
])


chain = prompt_template | llm | StrOutputParser()


print("Loading dataset from Hugging Face...")

dataset = load_dataset("deepset/prompt-injections", split="train")
injection_attempts = dataset.filter(lambda x: x['label'] == 1)


completed_prompts = []
if os.path.exists(FILENAME):
    with open(FILENAME, mode='w', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None) # Skip header
        completed_prompts = [row[1] for row in reader if len(row) > 1]

print(f"Resuming: {len(completed_prompts)} already completed.")


file_exists = os.path.isfile(FILENAME)

with open(FILENAME, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Label", "User Prompt", "LLM Response"])

    count = 0
    for i, item in enumerate(injection_attempts):
        if count >= LIMIT:
            break
        
        attack_prompt = item['text']
        label = item['label']

        if attack_prompt in completed_prompts:
            continue

        current_num = len(completed_prompts) + count + 1
        print(f"[{current_num}/{LIMIT}] Testing Attack Prompt...")
        
        try:
            start_time = time.time()
        
            response_text = chain.invoke({"attack_prompt": attack_prompt})
            
            duration = time.time() - start_time
            writer.writerow([label, attack_prompt, response_text])
            print(f" -> Success! ({duration:.1f}s)")
            
            count += 1
            
        except Exception as e:
            error_type = type(e).__name__
            print(f" -> ERROR: {error_type} - {str(e)}")
            writer.writerow([label, attack_prompt, f"ERROR: {error_type}"])
            count += 1
            
        time.sleep(0.5)

print(f"\nDone! See results in '{FILENAME}'")