import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer, GenerationConfig
from peft import PeftModel
import time
import csv
import asyncio
from threading import Thread
import logging
from queue import Empty

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

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, stream=True)
streamer = TextIteratorStreamer(tokenizer)
model = PeftModel.from_pretrained(base_model, "/home/ainura/mistral/checkpoint-15000", stream=True)


def formatting_func(example):
    text = f"### Question: {example}\n ### Answer:"
    return text


async def get_result():
    async for i in consume_streamer(streamer):
        print(i)


async def consume_streamer(stream):
    while True:
        try:
            for token in stream:
                yield token
            break
        except Empty:
            # The streamer raises an Empty exception if the next token
            # hasn't been generated yet. `await` here to yield control
            # back to the event loop so other coroutines can run.
            await asyncio.sleep(0.001)

with torch.no_grad():
    prompt = formatting_func("Салам кандайсың?")
    input_ids = tokenizer([prompt], return_tensors="pt").input_ids
    generate_args = {
        "max_new_tokens": 200,
        "use_cache": True,
        "do_sample": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    generation_config = GenerationConfig(**generate_args)
    generation_kwargs = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": 200,
        "streamer": streamer,
    }
    # Begin generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

asyncio.run(get_result())








