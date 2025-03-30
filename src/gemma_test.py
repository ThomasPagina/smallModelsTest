from transformers import pipeline
import torch
from huggingface_hub import login
import os
from dotenv import load_dotenv
# This is a test script for the Gemma model using Hugging Face's transformers library.

# gated model, so you need to authenticate with your Hugging Face token
load_dotenv()

# this one has to be set in the environment variables
# HUGGINGFACETOKEN=your_token
hugging_face_token = os.environ.get('HUGGING_FACE')



if not hugging_face_token:
    raise ValueError("HUGGINGFACETOKEN environment variable is not set.")
print(hugging_face_token)

login(token=hugging_face_token)

def load_prompts(directory: str) -> list[str]:
    """
    Load prompts from a given directory.
    """
    prompts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r",encoding="utf-8") as file:
                prompts.append(file.read())
    return prompts

load_prompts("prompts")
    
def execute_prompt(prompt: str) -> str:
    """
    Execute a prompt and return the response.
    """
    # Here you would implement the logic to execute the prompt
    # For this example, we'll just return the prompt itself
    
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2-2b-it",
        model_kwargs={"torch_dtype": torch.bfloat16},
        #device="cuda",  # replace with "mps" to run on a Mac device
    )

    messages = [
        {"role": "user", "content": prompt},
    ]

    outputs = pipe(messages, max_new_tokens=512)
    return outputs[0]["generated_text"][-1]["content"].strip()

for prompt in load_prompts("prompts"):
    print(f"Prompt: {prompt}")
    response = execute_prompt(prompt)
    print(f"Response: {response}")
    print("-" * 40) 