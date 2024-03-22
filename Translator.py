from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline, Conversation, TextIteratorStreamer
from threading import Thread
from sys import stdout
import time, os
from json import load as jload

config = jload(open("config.json"))
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=config['pretrained_model_name_or_path'])
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=config['pretrained_model_name_or_path'], config=model_config, device_map=config['device'])
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=config['pretrained_model_name_or_path'], config=model_config)
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

class Jade:
	def __init__(self, config, model_config, model, tokenizer, streamer):
		self.config = config
		self.model_config = model_config
		self.model = model
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
                output = llm.model.generate(llm.content_tokenized.input_ids, eos_token_id=llm.tokenizer.eos_token_id, pad_token_id=llm.tokenizer.pad_token_id, max_new_tokens=llm.config['max_new_tokens'], do_sample=llm.config['do_sample'], streamer=llm.streamer, temperature=llm.config['temperature'], top_k=llm.config['top_k'])

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

llm = Jade(config, model_config, model, tokenizer, streamer)

while True:
	time.sleep(0.2)
	llm.input = input("File Name>")
	llm.original_content = open('./content/'+llm.input+'.txt', 'r', encoding="utf-8").read()
	llm.translated_content = open('./content/'+llm.input+'-2.txt', 'r', encoding="utf-8").read()
	llm.content = tokenizer.apply_chat_template([{"role": "system", "content": llm.config['system_prompt']},{"role": "user", "content": "Translated:{"+llm.translated_content+"}Original:{"+llm.original_content+"}"}], tokenize=False, add_generation_prompt=True)
	llm.content_tokenized = tokenizer([llm.content], return_tensors="pt")
	generate(llm)
	llm.gen_proc.join()
