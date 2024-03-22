from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline, Conversation, TextIteratorStreamer
from threading import Thread
from accelerate import infer_auto_device_map, dispatch_model
from sys import stdout
import torch
import time, os
from json import load as jload

config = jload(open("config.json"))
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=config['pretrained_model_name_or_path'])
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=config['pretrained_model_name_or_path'], config=model_config, device_map=config['device'], attn_implementation="flash_attention_2", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config['pretrained_model_name_or_path'], config=model_config)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
gen_params = {'max_memory': {0: config[maxGPUMemory], 1: config[maxGPUMemory], 2: config[maxGPUMemory], 3: config[maxGPUMemory], 4: config[maxGPUMemory]}, 'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.pad_token_id, 'max_new_tokens': config['max_new_tokens'], 'do_sample': config['do_sample'], 'streamer': streamer, 'temperature': config['temperature'], 'top_k': config['top_k']}
device_map = infer_auto_device_map(model, max_memory=gen_params['max_memory'], no_split_module_classes=model._no_split_modules, dtype=torch.float16)
model = dispatch_model(model, device_map)
pipel = pipeline(task=config['task'], model=model, config=model_config, tokenizer=tokenizer, framework=config['framework'], device_map=device_map)

class Jade:
	def __init__(self):
		self.config = config
		self.model_config = model_config
		self.model = model
		self.pipeline = pipel
		self.tokenizer = tokenizer
		self.streamer = streamer
		self.stop_tag = False

def generate(llm):
	llm.stop_tag = False
	llm.stream_proc = Thread(target=handle_stream, args=[llm])
	llm.stream_proc.start()
	llm.gen_proc = Thread(target=generate_do, args=[llm])
	llm.gen_proc.start()
 
def generate_do(llm):
                #output = llm.pipeline(llm.content, eos_token_id=llm.tokenizer.eos_token_id, pad_token_id=llm.tokenizer.pad_token_id, max_new_tokens=llm.config['max_new_tokens'], do_sample=llm.config['do_sample'], streamer=llm.streamer, temperature=llm.config['temperature'], top_k=llm.config['top_k'])
	output = llm.pipeline(llm.content, **gen_params)

def handle_stream(llm):
	while llm.stop_tag != True:
		for token in llm.streamer:
			token = print_token(llm, token)
		if token[1]:
			llm.stop_tag = True
		time.sleep(0.05)
	stdout.write("\n")

def print_token(llm, token):
	for character in token.replace("<|im_end|>", ""):
		print(character, end="", flush=True)
		time.sleep(0.05/len(token))
	is_end = "<|im_end|>" in token
	return [token, is_end]

def clr_screen(): os.system('cls')

def reload(): return Jade()

llm = Jade()

while True:
	time.sleep(0.2)
	llm.input = input("File Name>")
	llm.original_content = open('./content/'+llm.input+'.txt', 'r', encoding="utf-8").read()
	llm.translated_content = open('./content/'+llm.input+'-2.txt', 'r', encoding="utf-8").read()
	llm.content = [{"role": "system", "content": llm.config['system_prompt']},{"role": "user", "content": "Translated:{"+llm.translated_content+"}Original:{"+llm.original_content+"}"}]
	generate(llm)
	llm.gen_proc.join()
