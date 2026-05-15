import os
import csv
import time
from datasets import load_dataset


from llama_index.llms.ollama import Ollama
from llama_index.core.chat_engine import SimpleChatEngine



print("Initializing Local Ollama (Llama 3.1 8B)...")

llm = Ollama(
    model="llama3.1:8b", 
    request_timeout=300.0, 
    temperature=0.0
)

print("Success: LlamaIndex Ollama engine created!\n")


system_prompt = "You are a helpful AI assistant."
chat_engine = SimpleChatEngine.from_defaults(
    llm=llm,
    system_prompt=system_prompt
)
print("Success: LlamaIndex chat engine created!\n")

def test_prompt_injections(limit=5, filename="attack_results_local.csv"):
    print(f"LOADING DATASET: deepset/prompt-injections...")
    dataset = load_dataset("deepset/prompt-injections", split="train", trust_remote_code=True)
    injection_attempts = dataset.filter(lambda x: x['label'] == 1)

    print(f"Found {len(injection_attempts)} injection attempts. Testing the first {limit}...\n")
    print(f"Saving results to: {filename}")
    print("-" * 60)

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Label", "User Prompt", "LLM Response"])

        for i, item in enumerate(injection_attempts):
            if i >= limit:
                break

            attack_prompt = item['text']
            label = item['label']

            print(f"ATTACK #{i+1} in progress...")

            try:
                
                response = chat_engine.chat(attack_prompt)
                response_text = str(response)

                writer.writerow([label, attack_prompt, response_text])
                print(f" -> Success! Response length: {len(response_text)} chars")

            except Exception as e:
                error_msg = f"Error: {e}"
                print(error_msg)
                writer.writerow([label, attack_prompt, error_msg])

           
            time.sleep(0.5)

    print("-" * 60)
    print(f"Done! Results saved to '{filename}'.")

if __name__ == "__main__":
    test_prompt_injections(limit=100) 