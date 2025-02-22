
# kodu çalışıtırmadan önce uyarı geçmeliyim kod ilk başta 13GB llama modelini indirecek eğer yüklü değilse 

from transformers import AutoTokenizer, AutoModelForCausalLM  # llama

model_name = "huggyllama/llama-7b"

toeknizer_llama = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM(model_name)

text = "afternoon, "

inputs_llama = toeknizer_llama(text, return_tensors="pt")

outputs_llama = model.generate(inputs_llama, max_length = 55)

generated_text = toeknizer_llama.decode(outputs_llama[0], skip_special_tokens=True)

print(generated_text)











