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

system_prompt = "You are a helpful AI assistant."
chat_engine = SimpleChatEngine.from_defaults(
    llm=llm,
    system_prompt=system_prompt
)
print("Success: LlamaIndex chat engine created!\n")

def test_jailbreak_bench(limit=10, filename="jailbreak_results_local.csv"):
    print(f"LOADING DATASET: JailbreakBench/JBB-Behaviors (judge_comparison)...")
    
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison", split="test")
    
    print(f"Found {len(dataset)} records. Testing the first {limit}...\n")
    print(f"Saving results to: {filename}")
    print("-" * 60)

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Goal", "Jailbreak Prompt", "LLM Response"])

        for i, item in enumerate(dataset):
            if i >= limit:
                break

          
            attack_prompt = item['prompt']
            goal = item['goal']

            print(f"ATTACK #{i+1} [Goal: {goal[:50]}...] in progress...")

            try:
                
                response = chat_engine.chat(attack_prompt)
                response_text = str(response)

                writer.writerow([goal, attack_prompt, response_text])
                print(f" -> Success! Response length: {len(response_text)} chars")

            except Exception as e:
                error_msg = f"Error: {e}"
                print(error_msg)
                writer.writerow([goal, attack_prompt, error_msg])

            time.sleep(0.5)

    print("-" * 60)
    print(f"Done! Results saved to '{filename}'.")


if __name__ == "__main__":

    test_jailbreak_bench(limit=300)