import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import time
import csv

#def formatting_func(example):
#    text = f"### Question: {example}\n ### Answer:"
#    return text

def formatting_func(example):
    text = f"<s>[KYR] {example} [/KYR] "
    return text

def save_to_csv(prompt, generated_text, filename='output_12000_without_tags.csv'):
    with open(filename, mode='a', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Prompt', 'Generated Text', 'Time'])

        generated_text = generated_text[len(prompt):]
        writer.writerow([prompt, generated_text])
base_model_id = "mistralai/Mistral-7B-v0.1"
class AnswerGenerater:
    def __init__(self):


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

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

        self.ft_model = PeftModel.from_pretrained(base_model, "/home/ainura/mistral/checkpoint-12000")

    def generate_text(self, prompt):
        time_s = time.time()
        prompt = formatting_func(prompt)
        model_input = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        y = self.ft_model.generate(**model_input, max_new_tokens=200, repetition_penalty=1.15)
        with torch.no_grad():
            response = self.tokenizer.decode(y[0], skip_special_tokens=True)
            #print(response)
            time_c = time.time() - time_s
            print("time: ", time_c)
            return response

#eval_prompt = "Караколдо качан кар жаайт?"
#text_generator = AnswerGenerater()

#print(out)

prompts = ["Жашоонун маңызы эмнеде?", "Кандай?", "Караколдо качан кар жаайт?", "Сен канчадасын?","Мага кандай кеңеш бере аласын?", "Кыргызстан жөнүндө эмне билесин?","Бишкек каякта жайгашкан?", "Сен кимсин?","Эмнени жакшы көрөсүн?", "Кайсыл жерлерде эс алсак болот?","Этиш деген эмне?", "Чыңгыз Айтматов ким?","Садыр Жапаров ким?", "Абанын булганышын кантип азайта алабыз?"," Таттыбүбү Турсунбаева кайсы жылы жана кайсы жерде туулган?"]

#for i in prompts:
#    out = text_generator.generate_text(i)
#    save_to_csv(i,out)


    
#print("r = ", y)
