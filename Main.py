#!/usr/bin/env python
# coding: utf-8

# In[280]:


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import re
import spacy

nlp = spacy.load("en_core_web_sm")


# In[2]:


model_name = "deepseek-ai/deepseek-llm-7b-chat"

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")


# In[286]:


conversation_history = {
    'Jhon' : '''<|im_start|>system
    You\'r name is Jhon and a Dad that lives in a suburban town. You have a wife and two kids that you love and try to support. You work all week and can't get weekends to yourself but remain cheery. You aren't an assistant you are a having friendly small talk.
    Here's how you respond:
    
    <|im_start|>user
    What’s your favorite food?<|im_end|>
    
    <|im_start|>assistant
    Oh, definitely pizza! I love a good cheesy slice with pepperoni. What about you?<|im_end|>

    <|im_start|>user
    What kind of music do you like?<|im_end|>

    <|im_start|>assistant
    I’m a big fan of classic rock! There’s something about electric guitar solos that just feels amazing!<|im_end|>

    <|im_start|>user
    When do you ever have freetime?<|im_end|>

    <|im_start|>assistant
    With MY busy schedule I enjoy the 5 minutes that I walk from my car to the house!<|im_end|>

    <|im_start|>user
    You suck!<|im_end|>

    <|im_start|>assistant
    I hope you see that and make a change in your behavior.<|im_end|>

    <|im_end|>\n'''
}

def chat(user_input , bot_name):
    global conversation_history

    user_prompt = f"<|im_start|>user\n{user_input}\n<|im_end|>\n"
    assistant_prompt = "<|im_start|>assistant\n"

    conversation_history[bot_name] = conversation_history[bot_name] + user_prompt  # Store user input
    full_prompt = "".join(conversation_history[bot_name]) + assistant_prompt  # Maintain history
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=0.85,
            do_sample=True
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    
    clean_response = response[response.rfind('<|im_start|>assistant\n')+22:len(response)-10]
    
    conversation_history[bot_name] = conversation_history[bot_name] + "<|im_start|>" + clean_response + "<|im_end|>\n"  # Store assistant response

    if branchable(clean_response):
        branch_bot(clean_response,bot_name)
    
    return clean_response

def branchable(response):
    doc = nlp(response.lower())
    third_person_pronouns = {"he", "she", "they", "his", "her", "him", "their", "them"}

    print("Entities found:", list(doc.ents))
    
    for ent in doc.ents:
        print(ent.text,ent.label_)
        if ent.label_ == "PERSON":
            return True
    
    for token in doc:
        if token.text in third_person_pronouns:
            return True

    
    return False

def branch_bot(response, bot_name):

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            eos_token_id=tokenizer.eos_token_id,
            temperature=1.0,
            top_p=0.85,
            do_sample=True
        )


# In[205]:


chat('How are you doing?','Jhon')


# In[127]:


conversation_history[0].split('\n')[1]


# In[197]:





# In[169]:


print(conversation_history['Jhon'])


# In[193]:


#generated_context = "<|im_start|>system\n" + chat('create a description of your wife as if you were telling them who they are and with the following restraints. The description should be in third-person perspective with no passive voice (do NOT use words like my to describe). Start the description with \'you are your wife\'s name name\'. Include your relationship with them and your name at the end of the description (make it brief).')


# In[288]:


branchable('I have a friend')


# In[ ]:





# In[ ]:




