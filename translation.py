from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Load the fine-tuned model and tokenizer
model_de = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
tokenizer_de = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
model_bg = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-bg-en")
tokenizer_bg = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-bg-en")

# Create a translation pipeline
translation_pipeline_de_en = pipeline("translation_de_to_en", model=model_de, tokenizer=tokenizer_de)
translation_pipeline_bg_en = pipeline("translation_bg_to_en", model=model_bg, tokenizer=tokenizer_bg)


# Function to translate German text to English
def translate_german_to_english(german_text):
    translated_text = translation_pipeline_de_en(german_text)
    return translated_text[0]['translation_text']


# Function to translate Bulgarian text to English
def translate_bulgarian_to_english(bulgarian_text):
    translated_text = translation_pipeline_bg_en(bulgarian_text)
    return translated_text[0]['translation_text']
