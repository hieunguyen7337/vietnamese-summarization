from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./envit5_model")  
model = AutoModelForSeq2SeqLM.from_pretrained("./envit5_model")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def whisper_predict():

    # read input text
    input = request.data.decode('utf-8')

    input_text =  "vietnews: " + input + " </s>"
    
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

    outputs = model.generate(input_ids=input_ids,
                            max_length=256,
                            early_stopping=True)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return output_text

if __name__ == '__main__':
    app.run(port=5000)