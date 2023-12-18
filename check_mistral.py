import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import time


base_model_id = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

ft_model = PeftModel.from_pretrained(base_model, "/home/ulan/mistral/mistral-nov-finetune/checkpoint-14000")
ft_model.eval()
eval_prompt = "Караколдо качан кар жаайт?"
while True:
    prompt = input("question = ")
    model_input = tokenizer(prompt, return_tensors="pt").to("cuda")


    y = ft_model.generate(**model_input, max_new_tokens=200, repetition_penalty=1.15)
#print("r = ", y)
    with torch.no_grad():
        print(tokenizer.decode(y[0], skip_special_tokens=True))

