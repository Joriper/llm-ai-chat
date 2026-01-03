from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM
from scrapper import scrapper
from peft import PeftModel, PeftConfig
import faiss,json
import numpy as np
from difflib import get_close_matches
import torch



#using model which generate a response of user query when pass the input
#model_name = "microsoft/DialoGPT-small"

text_generate_model = None
tokenizer = None
embedding = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def init_models():
    global text_generate_model, tokenizer, embedding

    if text_generate_model is not None:
        return  # already loaded ✅

    print("🔥 Loading models ONCE...")

    model_path = "./my_lora_model"

    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        peft_config.base_model_name_or_path,
        cache_dir="F:/hf_cache"
    )

    text_generate_model = PeftModel.from_pretrained(
        base_model,
        model_path
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path,
        cache_dir="F:/hf_cache"
    )

    embedding = SentenceTransformer(
        "all-MiniLM-L6-v2",
        cache_folder="F:/hf_cache"
    )

    print("✅ Models loaded successfully")

def answer(input_text,conversation_memory):
    MAX_MEMORY = 10
    MAX_INPUT_TOKEN = 50
    MAX_OUTPUT_TOKEN = 100
    TOTAL_TOKEN = MAX_INPUT_TOKEN+MAX_OUTPUT_TOKEN
    
    docs_data = scrapper(input_text)

    start,batch = 0,200


    with open("dataset.json", "r") as f:
        data = json.load(f)
    
    fine_inputs = [i['input'].lower() for i in data]

    def is_fine_tuning(input_text):
        matches = get_close_matches(input_text, fine_inputs, n=1, cutoff=0.5)
        return matches[0] if matches else None

    while start < len(docs_data):
        end = start+batch
        docs = docs_data[start:end]

        embed_doc = embedding.encode(docs)


        d = embed_doc.shape[1]          # embedding dimension
        index = faiss.IndexFlatL2(d)    # creating a tempory space which handle calculation of all the knowlege based on the user query.

        index.add(np.array(embed_doc))   # adding all the knowlege to the space. all the user query calculate the which is best


        print("Welcome to AI Assistant Thanks for using")

        check_length = len(tokenizer(input_text)['input_ids'])
        if check_length > MAX_INPUT_TOKEN:
            return "Please give shorter query."


        conversation_memory.append("Query: {}".format(input_text))

        last_top_conversation = 3

        #convert the user query to embedding which match the calculation based on the index knowlege added
        embedded_query = embedding.encode([input_text])

        distance,indices = index.search(np.array(embedded_query),last_top_conversation)  # serching number of the most important result. it use last_top_conversation for it

        max_distance = 1.5
        embed_result = [docs[i] for d,i in zip(distance[0],indices[0]) if d <= max_distance]

        if len(embed_result)  > 0:

            customize_for_conversation = "\n".join(embed_result)
            conversation = "\n".join(conversation_memory[-10:])
            result = is_fine_tuning(input_text)
            if result:
            
                user_query_modified = f"""
                    You are a trained assistant. Answer from fine-tuning dataset and you can customized the behavior and can use context also.
                    Do not summarize or simplify. Use only the most relevant context to answer the query and give perfect answer.

                    Conversation:
                        {conversation}
                    Context:
                        {customize_for_conversation}
                    Query:
                        {input_text}
                    Answer:
                """    
            else:

                user_query_modified = f"""
                    You are robot
                    Give the relevant answer only of query by using Context in 4 lines.

                    Conversation:
                        {conversation}
                    Context:
                        {customize_for_conversation}
                    Query:
                        {input_text}
                    Answer:
                """
            convert_query_like_chat_in_token = tokenizer(user_query_modified,return_tensors="pt").to(device)
            
            outputs = text_generate_model.generate(**convert_query_like_chat_in_token, max_new_tokens=MAX_OUTPUT_TOKEN, do_sample=False,repetition_penalty=1.2).to(device)
            actual_output = tokenizer.decode(outputs[0],skip_special_tokens=True)
            
            conversation_memory.append(f"Answer: {actual_output}")
            conversation_memory = conversation_memory[-MAX_MEMORY:]            
            return actual_output
        else:
            start+=batch
        
    else:            
        return "Sorry i don't have answer of your query!"
