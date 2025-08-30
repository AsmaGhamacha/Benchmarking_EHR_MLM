from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "microsoft/BioGPT"
save_path = "data/hf_corpora/microsoft__BioGPT"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_path)

model = AutoModelForCausalLM.from_pretrained(model_id)
model.save_pretrained(save_path)