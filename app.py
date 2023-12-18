import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/api/receive_data', methods=['POST'])
def receive_data():
    received_data = request.json  # Assuming the data is sent as JSON
    #print(received_data)
    #eval_prompt = "Майкл Джексон ким?"
    eval_prompt = received_data['in_text']
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
    	print(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True))
    	out = tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100, repetition_penalty=1.15)[0], skip_special_tokens=True)
    	print(out)
    response_data = {"out_text": out}
    return jsonify(response_data)

if __name__ == '__main__':
    base_model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)
    ft_model = PeftModel.from_pretrained(base_model, "./mistral-nov-finetune/checkpoint-13000")
    ft_model.eval()
    app.run(host='127.0.0.1', port=3050, debug=True)
    
    



