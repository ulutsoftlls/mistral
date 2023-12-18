import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
import transformers
from trl import SFTTrainer


train_dataset = load_dataset('json', data_files='./data/gpt4manual/train.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='./data/gpt4manual/validation.jsonl', split='train')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='right', add_eos_token=True)

def get_completion(query: str, model, tokenizer) -> str:
  device = "cuda:0"

  prompt_template = """
  <s>
  [INST]
  Below is an instruction that describes a task. Write a response that appropriately completes the request.
  {query}
  [/INST]
  </s>
  <s>

  """
  prompt = prompt_template.format(query=query)

  encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

  model_inputs = encodeds.to(device)


  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])
  
result = get_completion(query="code the fibonacci series in python using reccursion", model=model, tokenizer=tokenizer)
#print(result)

def formatting_func(example):
    prefix_text = 'Төмөндө тапшырманы сүрөттөгөн көрсөтмө келтирилген. Жооп жазуу ' \
    			  'бул өтүнүчтү орундуу түрдө аткарат.\n\n'
    text = f"<s>[INST]{prefix_text} {example['input']} [/INST] </s> \\n <s> {example['output']} </s>"
    return text

train = [formatting_func(data_point) for data_point in train_dataset]
train_dataset = train_dataset.add_column("prompt", train)
    
evald = [formatting_func(data_point) for data_point in eval_dataset]
eval_dataset = eval_dataset.add_column("prompt", evald)

#print(train_dataset["prompt"])

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

print(model)

def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)
  
  
modules = find_all_linear_names(model)
print(modules)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,#64
    lora_alpha=32,#16
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()

tokenizer.padding_side = 'right'

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="prompt",
    peft_config=lora_config,
    max_seq_length=2048,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=0.03,
        max_steps=60000,
        learning_rate=1e-4,
        logging_steps=2000,
        output_dir="./gpt4manual-output",
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=2000,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()


