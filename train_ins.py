from datasets import load_dataset
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

train_dataset = load_dataset('json', data_files='./data/all_translated_data/train.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='./data/all_translated_data/validation.jsonl', split='train')

class Config:
  def __init__(self):
    self.base_model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
    self.new_model_id = 'new-mistral-instruct'

    self.eval_percentile = 0.01
    self.max_token_length = 4096

    self.lora_r = 16
    self.lora_alpha = 32
    self.lora_dropout = 0.05 # 0.01
    self.lora_bias = "none" # Will set to "all" for better quality but slower training (switching causes runtime error)
    self.lora_task_type = "CAUSAL_LM"


    total_steps = 1000
    dataset_size = 260000

    self.batch_size = 2
    self.eval_batch_size = 1
    self.train_steps = (dataset_size // self.batch_size)
    self.warmup_steps = int(self.train_steps * 0.1)
    self.gradient_accumulation_steps = 1
    self.learning_rate = 2e-4
    self.use_bf16 = True
    self.weight_decay = 0.001

    self.checkpoint_step = self.train_steps // total_steps
    self.response_template = "### Answer:"

config = Config()

def formatting_func(example):
    text = f"[INST] {example['instruction']}\n [/INST] ### Answer: {example['output']} #End"
    return text

def plot_data_lengths(train_dataset, eval_dataset):
    """ Check the distribution of the lengths of the tokenized prompts """

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_id,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_prompt_wo_pad(prompt):
        return tokenizer(prompt['prompt'])

    tokenized_train_dataset = train_dataset.map(tokenize_prompt_wo_pad)
    tokenized_eval_dataset = eval_dataset.map(tokenize_prompt_wo_pad)

    lengths = [len(x['input_ids']) for x in tokenized_train_dataset]
    lengths += [len(x['input_ids']) for x in tokenized_eval_dataset]
    print(len(lengths))

    # Plotting the histogram
    #plt.figure(figsize=(10, 6))
    #plt.hist(lengths, bins=20, alpha=0.7, color='blue')
    #plt.xlabel('Length of input_ids')
    #plt.ylabel('Frequency')
    #plt.title('Distribution of Lengths of input_ids')
    #plt.show()

def test_model(model, prompt):
  tokenizer = AutoTokenizer.from_pretrained(
      config.base_model_id,
      add_bos_token=True,
  )

  model_input = tokenizer(prompt, return_tensors="pt").to("cuda")

  model.eval() # switches the model into evaluation mode, disables things like dropout
  with torch.no_grad():
    response = tokenizer.decode(model.generate(**model_input, max_new_tokens=256, repetition_penalty=1.15)[0], skip_special_tokens=True)

  return response

def print_stats(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Add the begin and end sequence tokens
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_id,
    padding_side="right",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = [*map(formatting_func, train_dataset)]
eval_dataset = [*map(formatting_func, eval_dataset)]



train_dataset = Dataset.from_dict({"prompt": train_dataset})
eval_dataset = Dataset.from_dict({"prompt": eval_dataset})

print(f'Train: {len(train_dataset)} to eval: {len(eval_dataset)}')

# The <s> and </s> are tokens wrapping a logical unit of the context, it ususally consist of instruction - answer pairs
# The [INST] and [/INST] token wraps an instruction
# Read more: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
def log_tokenized(*texts):
    l = max(len(text) for text in texts)

    print("Tokenized")
    print('\n'.join(f'{text.ljust(l)}:{tokenizer(text).input_ids}' for text in texts))
    print("\nPartialy decoded")
    print('\n'.join(f'{text.ljust(l)}:{tokenizer.decode(tokenizer(text).input_ids, skip_special_tokens = True)}' for text in texts))

log_tokenized(
  "кандай",
  "<s> кандай </s>",
  "[ INST ] кандай",
  "[INST] кандай",
  "[INST] кандай [/INST]",
)

print("\ntokenizer")


# About the terminology and the parameters: https://aisuko.gitbook.io/wiki/ai-techniques/training/training-with-qlora
bnb_config = BitsAndBytesConfig(
    # Blogpost about LLama 2 7b inference speed https://openmmlab.medium.com/faster-and-more-efficient-4-bit-quantized-llm-model-inference-a27d35a66c29
    # Research paper about k bit quantization inference https://arxiv.org/pdf/2212.09720.pdf
    # The conclusion is if the quantized nodes are frozen 4 bit quantization is the best
    load_in_4bit=True,

    # The name of the technique proposed by the same qlora paper
    bnb_4bit_use_double_quant=True,

    # Use the "normal float" quantization
    bnb_4bit_quant_type="nf4",

    # The datatype that is used during algebratic operations
    bnb_4bit_compute_dtype=torch.bfloat16
)



lora_config = LoraConfig(
    # A very detailed comperison of different r and alpha values: https://github.com/cloneofsimo/lora/discussions/37
    r=config.lora_r,
    lora_alpha=config.lora_alpha,

    # The targeted matrixes for LoRA are taken from: https://brev.dev/blog/fine-tuning-mistral-your-own-data#4-set-up-lo-ra
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],

    bias=config.lora_bias,
    lora_dropout=config.lora_dropout,
    task_type=config.lora_task_type,
)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_id,
    use_cache=False,
    quantization_config=bnb_config,
)


# Prepares the model for kbit training by doing the following:
# Cast the layernorm in fp32
# Making output embedding layer require grads
# Add the upcasting of the lm head to fp32
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

model = get_peft_model(model, lora_config)

# Gradient checkpointing: https://residentmario.github.io/pytorch-training-performance-guide/gradient-checkpoints.html
# TLDR a memory / speet tradeof: https://residentmario.github.io/pytorch-training-performance-guide/gradient-checkpoints.html#benchmarks
model.gradient_checkpointing_enable({"use_reentrant": True})

model.print_trainable_parameters()

tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

training_args = TrainingArguments(
    # Directory to put data to
    output_dir='train_ins_mistral',

    # the number of steps in the training, where
    # each step is an evluation of a batch
    max_steps=config.train_steps,

    # the number of steps with small, and increasing learning rate
    # in the end the target learning rate is hit, after that the
    # learning rate scheduler takes over
    # this allows the adaptive optimizer to gather statistics
    # and smooths out the start
    warmup_steps=config.warmup_steps,

    # the size of each batch, its optimal to tune it to fill up the available memory
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.eval_batch_size,
    # if there isn't enought memory for a large enought batch size for
    # batch normalization, then you can set this to signal the trainer
    # to only do the back propagation after k steps, this simulates
    # the behaviour of a larger batch size, but it is slower
    gradient_accumulation_steps=config.gradient_accumulation_steps,

    # The "starting" learning rate, after warmup
    learning_rate=config.learning_rate,

    # use bf 16 if possible (only on A100 or other Ampere seriese gpus)
    # read about it https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
    bf16=config.use_bf16,
    fp16=not config.use_bf16,

    # Paging the optimizers data to allow for larger models
    # Using adamw to allow for weight decay, to prevent overfitting
    # Use 32 bit floating point for internal calculations, paged_adamw_8bit also exists
    # During training the intermediate activation values and the gradients (if gradient checkpointing is not set)
    # are using the 16 bf or fp datatype
    # The optimizers states, such as moving averages, momentum etc, and the calculation of the loss function
    # and the gradiant accumulation if performed in 32 bit
    optim="paged_adamw_32bit",

    # provides regularization, and prevents the lower range bf16 numbers form projecting into
    # not representable numbers, read more: https://arxiv.org/pdf/2310.04415.pdfl
    # the paper proposes 0.1 as a good generay starting point
    weight_decay=config.weight_decay,

    # Directory for storing logs
    logging_dir="./logs",
    logging_steps=config.checkpoint_step,

    # Save the model checkpoint every logging step
    save_strategy="steps",
    save_steps=config.checkpoint_step,

    # Evaluate the model every logging step
    evaluation_strategy="steps",
    eval_steps=config.checkpoint_step,

    # Perform evaluation at the end of training
    do_eval=True,

    report_to="tensorboard",
)

# docs for the trainer: https://huggingface.co/docs/trl/main/en/sft_trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="prompt",
    args=training_args,
    # about the data collector: https://huggingface.co/docs/trl/main/en/sft_trainer#train-on-completions-only
    data_collator=DataCollatorForCompletionOnlyLM(config.response_template, tokenizer=tokenizer),
    packing=False,
    tokenizer=tokenizer,
    max_seq_length=config.max_token_length,
)

trainer.train()

