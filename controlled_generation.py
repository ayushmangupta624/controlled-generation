# -*- coding: utf-8 -*-
"""
Controlled Generation of Code-Mixed Text
=========================================
Implementation accompanying the paper:

  Multilingual Controlled Generation And Gold-Standard-Agnostic
  Evaluation of Code-Mixed Sentences
  Ayushman Gupta*, Akhil Bhogal*, Kripabandhu Ghosh
  IISER Kolkata
  https://arxiv.org/abs/2410.10580

This file is a direct conversion of the original Colab notebook to a
standalone Python script. No logic has been changed. The only differences
from the original notebook are:
  - API keys are read from environment variables instead of being hardcoded.
  - Section headers from the notebook are preserved as comments.

Usage:
  Set the following environment variables before running:
    OPENAI_API_KEY   - your OpenAI API key
    GEMINI_API_KEY   - your Google Gemini API key

  Then run individual sections as needed (the file is structured to mirror
  the original notebook cells).

Dataset paths (update these to match your local setup):
  Hindi   : Data/L3Cube_hing_twitter.txt
  Bengali : BN_Eng_data/FB_BN_EN_FN.txt
            BN_Eng_data/TWT_BN_EN_FN.txt
            BN_Eng_data/WA_BN_EN_FN.txt
  Spanish : Spa_Eng_data/mt_spanglisheng/spanglish.txt
"""

import pandas as pd
import glob
import os
import openai
import json
from collections import Counter
import re
import time

openai.api_key = os.environ.get("OPENAI_API_KEY", "")

def get_completion_from_messages_gpt4(messages, model="gpt-4", temperature=0, timeout=600):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, timeout=600):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout
    )
    return response.choices[0].message["content"]


import google.generativeai as genai
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
model = genai.GenerativeModel('gemini-pro')
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

"""DICTIONARY CREATION (USING TWITTER DATA)"""

# Here, we have loaded the L3Cube_Hing Dataset as mentioned in the paper. U may choose any social-media data as per your desire but may need to change the code.

with open("Data/L3Cube_hing_twitter.txt", 'r', encoding='utf-8') as txt_file:
    data5 = [(line.strip()).split() for line in txt_file]

data5_new=[]
alpha=f""""""
for i in range(len(data5)):
    if len(data5[i])==2:
        alpha=alpha+data5[i][0]+' '
    elif len(data5[i])==0:
        data5_new.append(alpha.strip())
        alpha=f""""""


data5_allWords=[]
data5_engWords=[]  #not necessary, just for stat.
for i in range(len(data5)):
    if len(data5[i])==2:
        data5_allWords.append(data5[i][0])
        if data5[i][1]=='EN':
            data5_engWords.append(data5[i][0])
    elif len(data5[i])==0:
        continue

data5_allDict={}
for word in data5_allWords:
    data5_allDict[word]=data5_allDict.get(word, 0) + 1

len(data5_allDict)

#English-Spanish
with open("Spa_Eng_data/mt_spanglisheng/spanglish.txt", 'r', encoding='utf-8') as txt_file:  # dataset mentioned in paper
    data1 = [line.strip() for line in txt_file]
overall_data=[]
for i in range(len(data1)):
    alpha=data1[i].split()
    for j in range(len(alpha)):
        overall_data.append(alpha[j].lower())
data_allDict={}
for word in overall_data:
    data_allDict[word]=data_allDict.get(word, 0) + 1

len(data_allDict)  # 29783  (despite small length the performance is even better than english-hindi)

#English-Bengali
with open("BN_Eng_data/FB_BN_EN_FN.txt", 'r', encoding='utf-8') as txt_file:
    data1 = [line.strip() for line in txt_file]
with open("BN_Eng_data/TWT_BN_EN_FN.txt", 'r', encoding='utf-8') as txt_file:
    data2 = [line.strip() for line in txt_file]
with open("BN_Eng_data/WA_BN_EN_FN.txt", 'r', encoding='utf-8') as txt_file:
    data3 = [line.strip() for line in txt_file]
data1_u=[]
for i in range(len(data1)):
        if len(data1[i].split())==3:
         data1_u.append((data1[i].split()[0]))
        if len(data1[i].split())<3:
             data1_u.append("BLANKKKK")

data2_u=[]
for i in range(len(data2)):
        if len(data2[i].split())==3:
         data2_u.append((data2[i].split()[0]))
        if len(data2[i].split())<3:
             data2_u.append("BLANKKKK")

data3_u=[]
for i in range(len(data3)):
        if len(data3[i].split())==3:
         data3_u.append((data3[i].split()[0]))
        if len(data3[i].split())<3:
             data3_u.append("BLANKKKK")

data1_updated=[]
for i in range(len(data1)):
        if len(data1[i].split())==3:
         data1_updated.append((data1[i].split()[0]).lower())

data2_updated=[]
for i in range(len(data2)):
        if len(data2[i].split())==3:
         data2_updated.append((data2[i].split()[0]).lower())

data3_updated=[]
for i in range(len(data3)):
        if len(data3[i].split())==3:
         data3_updated.append((data3[i].split()[0]).lower())

overall_data=data1_u+data2_u+data3_u


Overall_DATA=[]
temp_data=""
for i in range(len(overall_data)):
    if overall_data[i] not in ["BLANKKKK",'BLANKKKK']:
        temp_data=temp_data+overall_data[i]+" "
    else:
        Overall_DATA.append(temp_data.strip())
        temp_data=""

data_allDict={}
for word in overall_data:
    data_allDict[word]=data_allDict.get(word, 0) + 1


"""BASE CREATION USING LLM"""

Eng_sentt="As the first rays of dawn painted the sky in hues of pink and gold, the sleepy town came to life, welcoming a new day filled with possibilities and joys."

"""Prompt A  (needs GPT-4)"""

#MAIN PROMPT

prompt = f""" For the given English sentence, do the following:
1. Do PoS Tagging
2. For the words which are either  Noun (NN), Adjective (JJ), Adverb (RB), CC, or Interjection (UH), create a dictionary called 'Imp_Eng' with english words as main thing and
3. Translate the original english sentence into hindi
4. From the list 'Imp_Eng', look for the corresponding meaning in hindi and look them up in the hindi sentence. create a dictionary named 'hin_eng_dictionary'
5. transliterate each hindi word in 'hin_eng_dictionary' into Roman in three ways or spellings and add that in the dictionary.
6. format above as RFC8259 compliant json dictionary in the format ["eng":<eng word>,"pos_tag":<PoS Tag>, "hindi":<hindi word>, "roman_hindi": <transliterations>]

english sentence : {Eng_sentt},
"""
messages = [{"role": "user", "content": prompt}]

responsee = get_completion_from_messages_gpt4(messages,temperature=0,timeout=100)


#  english-bengali
prompt = f""" For the given English sentence, do the following:
1. Do PoS Tagging
2. For the words which are either  VB, Noun (NN), Adjective (JJ), Adverb (RB), CC, or Interjection (UH), create a dictionary called 'Imp_Eng' with english words as main thing and
3. Translate the original english sentence into Bengali
4. From the list 'Imp_Eng', look for the corresponding meaning in Bengali and look them up in the bengali sentence. create a dictionary named 'ben_eng_dictionary'
5. transliterate each bengali word in 'ben_eng_dictionary' into Roman in three ways or spellings and add that in the dictionary.
6. format above as RFC8259 compliant json dictionary in the format ["eng":<eng word>,"pos_tag":<PoS Tag>, "bengali":<bengali word>, "roman_bengali": <transliterations>]

english sentence : {Eng_sentt},
"""

# english-spanish
prompt = f""" For the given English sentence, do the following:
1. Do PoS Tagging
2. For the words which are either  Verb, Noun (NN), Adjective (JJ), Adverb (RB), CC, or Interjection (UH), create a dictionary called 'Imp_Eng' with english words as main thing and also include respective PoS Tags.
3. Translate the original english sentence into Spanish
4. From the list 'Imp_Eng', look for the corresponding meaning in Spanish and look them up in the spanish sentence. create a dictionary named 'spa_eng_dictionary'
5. transliterate each spanish word in 'spa_eng_dictionary' into Roman in three ways or spellings that can be found in social media or twitter and add that in the dictionary.
6. format above as RFC8259 compliant json dictionary in the format ["eng":<eng word>,"pos_tag":<PoS Tag>, "spanish":<spanish word>, "roman_spanish": <transliterations>]

english sentence : {Eng_sentt},
"""

# hindi sentence extraction
prompt = f"""You have been given LLM output(in triple ticks) in the format:
'''1. <text>
2. <text>
.
.
.
5.<text>
6.<text>'''
Extract the hindi sentence in the 3rd step without any tags or anything and make sure that there are no extra inverted commas or '/' sign. Just give the final sentence as output It shouldn't be like 'The final senntence is' or anything.

LLM Output : {responsee},
"""
messages = [{"role": "user", "content": prompt}]

responsee_hin = get_completion_from_messages_gpt4(messages,temperature=0,timeout=100)

#main base extraction
prompt = f"""You have been given LLM output(in triple ticks) in the format:
```1. <text>
2. <text>
.
.
.
5.<text>
6. Json Output <text>
```
Extract the json dictionary from '6. Json Output' and give output as json dictionary without any explanation or additional lines so that it can be used directly

LLM Output : {responsee},
"""
messages = [{"role": "user", "content": prompt}]
responsee_dict = get_completion_from_messages_gpt4(messages,temperature=0,timeout=100)

#main base dictionary cleaning
responsee_dict=(re.sub('\n+', '', responsee_dict))
#final dictionary
responsee_dict_format=json.loads(responsee_dict)
json_finals=[]
json_finals.append(responsee_dict_format)

# Base creation in one RUN with Prompt A
Overall=[]
Eng_sentt="Acceleration of water in this river is not due to gravity?"
prompt = f""" For the given English sentence, do the following:
1. Do PoS Tagging
2. For the words which are either  Noun (NN), Adjective (JJ), Adverb (RB), CC, or Interjection (UH), create a dictionary called 'Imp_Eng' with english words as main thing and
3. Translate the original english sentence into hindi
4. From the list 'Imp_Eng', look for the corresponding meaning in hindi and look them up in the hindi sentence. create a dictionary named 'hin_eng_dictionary'
5. transliterate each hindi word in 'hin_eng_dictionary' into Roman in three ways or spellings and add that in the dictionary.
6. format above as RFC8259 compliant json dictionary in the format ["eng":<eng word>,"pos_tag":<PoS Tag>, "hindi":<hindi word>, "roman_hindi": <transliterations>]

english sentence : {Eng_sentt},
"""
messages = [{"role": "user", "content": prompt}]

responsee = get_completion_from_messages_gpt4(messages,temperature=0,timeout=100)

prompt = f"""You have been given LLM output(in triple ticks) in the format:
'''1. <text>
2. <text>
.
.
.
5.<text>
6.<text>'''
Extract the hindi sentence in the 3rd step without any tags or anything and make sure that there are no extra inverted commas or '/' sign. Just give the final sentence as output It shouldn't be like 'The final senntence is' or anything.

LLM Output : {responsee},
"""
messages = [{"role": "user", "content": prompt}]

responsee_hin = get_completion_from_messages_gpt4(messages,temperature=0,timeout=100)

prompt = f"""You have been given LLM output(in triple ticks) in the format:
```1. <text>
2. <text>
.
.
.
5.<text>
6. Json Output <text>
```
Extract the json dictionary from '6. Json Output' and give output as json dictionary without any explanation or additional lines so that it can be used directly

LLM Output : {responsee},
"""
messages = [{"role": "user", "content": prompt}]
responsee_dict = get_completion_from_messages_gpt4(messages,temperature=0,timeout=100)

responsee_dict=(re.sub('\n+', '', responsee_dict))
responsee_dict_format=json.loads(responsee_dict)
json_finals=[]
json_finals.append(responsee_dict_format)
Overall.append({'Eng':Eng_sentt,'Hin':responsee_hin,'JSON_dict':responsee_dict_format})
k=0
Eng_sentt=Overall[k]['Eng']
responsee_hin=Overall[k]['Hin']
json_finals=[]
json_finals.append(Overall[k]['JSON_dict'])

# Base creation in one RUN with Prompt A for English-Spanish
Overall=[]

Eng_sentt=spa_eng_sent[i]
prompt = f""" For the given English sentence, do the following:
1. Do PoS Tagging
2. For the words which are either  Noun (NN), Adjective (JJ), Adverb (RB), CC, or Interjection (UH), create a dictionary called 'Imp_Eng' with english words as main thing and also include respective PoS Tags.
3. Translate the original english sentence into Spanish
4. From the list 'Imp_Eng', look for the corresponding meaning in Spanish and look them up in the spanish sentence. create a dictionary named 'spa_eng_dictionary'
5. transliterate each spanish word in 'spa_eng_dictionary' into Roman in three ways or spellings that can be found in social media or twitter and add that in the dictionary.
6. format above as RFC8259 compliant json dictionary in the format ["eng":<eng word>,"pos_tag":<PoS Tag>, "spanish":<spanish word>, "roman_spanish": <transliterations>]

english sentence : {Eng_sentt},
"""
messages = [{"role": "user", "content": prompt}]

responsee = get_completion_from_messages_gpt4(messages,temperature=0,timeout=100)

prompt = f"""You have been given LLM output(in triple ticks) in the format:
'''1. <text>
2. <text>
.
.
.
5.<text>
6.<text>'''
Extract the spanish sentence in the 3rd step without any tags or anything and make sure that there are no extra inverted commas or '/' sign. Just give the final sentence as output It shouldn't be like 'The final senntence is' or anything.

LLM Output : {responsee},
"""
messages = [{"role": "user", "content": prompt}]

responsee_spa = get_completion_from_messages_gpt4(messages,temperature=0,timeout=100)

prompt = f"""You have been given LLM output(in triple ticks) in the format:
```1. <text>
2. <text>
.
.
.
5.<text>
6. Json Output <text>
```
Extract the json dictionary from '6. Json Output' and give output as json dictionary without any explanation or additional lines so that it can be used directly

LLM Output : {responsee},
"""
messages = [{"role": "user", "content": prompt}]
responsee_dict = get_completion_from_messages_gpt4(messages,temperature=0,timeout=100)

responsee_dict=(re.sub('\n+', '', responsee_dict))
responsee_dict_format=json.loads(responsee_dict)
json_finals=[]
json_finals.append(responsee_dict_format)

Overall.append({'Eng':Eng_sentt,'Spa':responsee_spa,'JSON_dict':responsee_dict_format})


"""Prompt B (can run with Gemini Pro, GPT-3.5-Turbo as well along with GPT-4)"""

# Works with GPT-4 as well as Gemini Pro
prompt = f""" For the given English sentence, do the following:
    create this RFC8259 compliant json dictionary in the format ["hindi_trans": <hindi translation>,"Word_Dict":["eng":<eng word>,"base_eng":<base form of the english word>, "eng_pos_tag":<English PoS Tag>, "roman_hindi": <three different spellings of roman transliteration for hindi word>]] by
    doing PoS tagging of english sentence and then only choosing the words which are either Noun (NN), Adjective (JJ), Adverb (RB), CC, or Interjection (UH). And then translating the\
    english sentence into Hindi and then looking for the corresponding meaning of these english words in that. Also, for each hindi word, transliterate it into three different spellings that can be seen in twitter.
    The output should be RFC8259 compliant json dictionary without any additional words or description

    english sentence : {Eng_sentt},
    """

# change the language and the tags as per the choise. As for spanish and french, no need for transliteration.

# GPT-3.5-Turbo gets confused with no '{}' in the previous prompt. So, this works fine with it.
prompt = f""" For the given English sentence, do the following:
create this RFC8259 compliant json dictionary in the format {{"hindi_trans": <hindi translation>,"Word_Dict":[{{"eng":<eng word>,"base_eng":<base form of the english word>, "eng_pos_tag":<English PoS Tag>, "hindi":<hindi word>, "roman_hindi": <three different spellings of roman transliteration for hindi word>}}]}} by
doing PoS tagging of english sentence and then only choosing the words which are either Noun (NN), Adjective (JJ), Adverb (RB), CC, or Interjection (UH). And then translating the\
english sentence into Hindi and then looking ofr the corresponding meaning of these english words in that.
The output should be RFC8259 compliant json dictionary without any additional words or description

english sentence : {Eng_sentt},
"""
messages = [{"role": "user", "content": prompt}]

responsee2 = get_completion_from_messages(messages,temperature=0,timeout=100)

#this was the exact prompt used for the experiments in the paper (evaluation). the unwanted PoS tags were separated.
prompt = f""" For the given English sentence, do the following:
create this RFC8259 compliant json dictionary in the format ["hindi_trans": <hindi translation>,"Word_Dict":["eng":<eng word>,"base_eng":<base form of the english word>, "eng_pos_tag":<English PoS Tag>, "hindi":<hindi word>,"hin_pos_tag":<hindi PoS Tag>, "roman_hindi": <three different spellings of roman transliteration for hindi word>]] by
doing PoS tagging of english sentence and then only choosing the words which are either  Verb, Noun (NN), Adjective (JJ), Adverb (RB), CC, or Interjection (UH). And then translating the\
english sentence into Hindi and then looking for the corresponding meaning of these english words in that. and doing the PoS tagging of Hindi sentence as well and adding the\
respective hindi PoS tags(VB, VFN etc.) to the dictionary as well. Also, for each hindi word, transliterate it into three different spellings that can be seen in twitter.
The output should be RFC8259 compliant json dictionary without any additional words or description

english sentence : {Eng_sentt},
"""


response = model.generate_content(prompt,
                                safety_settings=safety_settings,
                                                    generation_config=genai.types.GenerationConfig(
                                candidate_count=1,
                                #stop_sequences=['.'],
                                #max_output_tokens=550,
                            #  top_p = 0.7,
                            # top_k = 4,
                                temperature=0))
responsee=response.text

# for extracting the output from Prompt B (no prompt need, just simply this)
def extract_text(s):
    first_bracket = s.find('{')
    last_bracket = s.rfind('}')
    if first_bracket != -1 and last_bracket != -1:
        return s[first_bracket:last_bracket+1]
    else:
        return ""

extracted_dict=extract_text(responsee)
dict_temp=json.loads(re.sub('\n+', '', extracted_dict))
responsee_hin=dict_temp['hindi_trans']

#the following seems somewhat necessary for Gemini Pro
# to cross check and remove the certain PoS tags we don't want that were included despite instructions (may add more)
word_dic_temp=[]
for k in range(len(dict_temp['Word_Dict'])):
    if (dict_temp['Word_Dict'][k]['eng_pos_tag'] not in ['VERB','Verb','VB','VBD','VBN','VFN','VBG','VBP','VBZ','PRP','PSP','TO','IN','DT','MD','PRP$']) & (dict_temp['Word_Dict'][k]['roman_hindi'] !=[])&(dict_temp['Word_Dict'][k]['eng_pos_tag'] !=None)&(dict_temp['Word_Dict'][k]['base_eng'] !=None)&(dict_temp['Word_Dict'][k]['hindi'] !=None):
            word_dic_temp.append(dict_temp['Word_Dict'][k])


json_finals.append(word_dic_temp)


"""GENERATING CODE-MIXED SENTENCES OF DIFFERENT CMD.   (NON-LLM PART)"""

# for english-spanish and english-french, do this step as well so as to also check the spanish word itself and not just its spelling variations.
for j in range(len(json_finals)):
    for i in range(len(json_finals[j])):
        if json_finals[j][i]['spanish'] not in json_finals[j][i]['roman_spanish']:
            json_finals[j][i]['roman_spanish'].append(json_finals[j][i]['spanish'])

# ASSIGNING SCORES

def replace_word(sentence, word_to_replace, replacement_word):
    # Use re.sub() to replace the word in the sentence
    pattern = re.escape(word_to_replace)
    new_sentence = re.sub(r'(?<!\w)' + pattern + r'(?!\w)', replacement_word, sentence)
    return new_sentence

eng_counts={}
hin_counts={}
sent_dict=[]
i=0
for j in range(len(json_finals[i])):
    try:
        eng_counts[((json_finals[i][j]['eng']))]=(data5_allDict[((json_finals[i])[j]['eng']).lower()])
    except KeyError:
        print("Eng Key not found: ",json_finals[i][j]['eng'])
        eng_counts[(json_finals[i][j]['eng'])]=0
    hin_count=0
    hin_temp={}
    for h in json_finals[i][j]['roman_hindi']:
        try:
            hin_count=hin_count+data5_allDict[h.lower()]
            hin_temp[h]=data5_allDict[h.lower()]
        except KeyError:
            print("Key not found: ",h.lower())
            hin_temp[h]=0
    hin_temp= (sorted(hin_temp.items(), key=lambda x: x[1]))[-1]

    hin_counts[hin_temp[0]]=hin_temp[1]
    if hin_temp[1]==0:
        score=0
    else:
        score=(eng_counts[(json_finals[i][j]['eng'])]/hin_temp[1])+0.1


    dict_temp={"eng":(json_finals[i][j]['eng']),"eng_score":eng_counts[(json_finals[i][j]['eng'])],"hin":json_finals[i][j]['hindi'],"hin_roman":hin_temp[0],"hin_score":hin_temp[1],"score":score}
    sent_dict.append(dict_temp)

# for including base form of English Words as well   (Works well with Prompt B)
Main_sent_dict=[]
for i in range(len(json_finals)):
    print(i)
    eng_counts={}
    hin_counts={}
    sent_dict=[]
    for j in range(len(json_finals[i])):
        try:
            enggg1=(data5_allDict[((json_finals[i])[j]['eng']).lower()])
        except KeyError:
            print("Eng Key not found: ",json_finals[i][j]['eng'])
            enggg1=0
        try:
            enggg2=(data5_allDict[((json_finals[i])[j]['base_eng']).lower()])
        except KeyError:
            print("Eng Base Key not found: ",json_finals[i][j]['base_eng'])
            enggg2=0
        eng_counts[((json_finals[i][j]['eng']))]=enggg1+enggg2
        hin_count=0
        hin_temp={}
        for h in json_finals[i][j]['roman_hindi']:
            try:
                hin_count=hin_count+data5_allDict[h.lower()]
                hin_temp[h]=data5_allDict[h.lower()]
            except KeyError:
                print("Key not found: ",h.lower())
                hin_temp[h]=0
        hin_temp= (sorted(hin_temp.items(), key=lambda x: x[1]))[-1]

        hin_counts[hin_temp[0]]=hin_temp[1]
        if hin_temp[1]==0:
            score=0
        else:
            score=(eng_counts[(json_finals[i][j]['eng'])]/hin_temp[1])+0.1


        dict_temp={"eng":(json_finals[i][j]['eng']),"eng_score":eng_counts[(json_finals[i][j]['eng'])],"hin":json_finals[i][j]['hindi'],"hin_pos":json_finals[i][j]['hin_pos_tag'],"hin_roman":hin_temp[0],"hin_score":hin_temp[1],"score":score}
        sent_dict.append(dict_temp)
    Main_sent_dict.append(sent_dict)

#english-bengali    (Prompt A; may need to modify for prompt B)
Main_sent_dict=[]

for i in range(len(json_finals)):
    eng_counts={}
    ben_counts={}
    sent_dict=[]
    for j in range(len(json_finals[i])):
        try:
            eng_counts[((json_finals[i][j]['eng']))]=(data_allDict[((json_finals[i])[j]['eng']).lower()])
        except KeyError:
            print("Eng Key not found: ",json_finals[i][j]['eng'])
            eng_counts[(json_finals[i][j]['eng'])]=0
        ben_count=0
        ben_temp={}
        for h in json_finals[i][j]['roman_bengali']:
            try:
                ben_count=ben_count+data_allDict[h.lower()]
                ben_temp[h]=data_allDict[h.lower()]
            except KeyError:
                print("Key not found: ",h.lower())
                ben_temp[h]=0
        ben_temp= (sorted(ben_temp.items(), key=lambda x: x[1]))[-1]

        ben_counts[ben_temp[0]]=ben_temp[1]
        if ben_temp[1]==0:
            score=0
        else:
            score=(eng_counts[(json_finals[i][j]['eng'])]/ben_temp[1])+0.1


        dict_temp={"eng":(json_finals[i][j]['eng']),"pos_tag":json_finals[i][j]['pos_tag'],"eng_score":eng_counts[(json_finals[i][j]['eng'])],"ben":json_finals[i][j]['bengali'],"hin_roman":ben_temp[0],"ben_score":ben_temp[1],"score":score}
        sent_dict.append(dict_temp)
    Main_sent_dict.append(sent_dict)

#english-spanish  (Prompt A; may need to modify for prompt B)


Main_sent_dict=[]

for i in range(len(json_finals)):
    eng_counts={}
    ben_counts={}
    sent_dict=[]
    for j in range(len(json_finals[i])):
        try:
            eng_counts[((json_finals[i][j]['eng']))]=(data_allDict[((json_finals[i])[j]['eng']).lower()])
        except KeyError:
            print("Eng Key not found: ",json_finals[i][j]['eng'])
            eng_counts[(json_finals[i][j]['eng'])]=0
        ben_count=0
        ben_temp={}
        for h in json_finals[i][j]['roman_spanish']:
            try:
                ben_count=ben_count+data_allDict[h.lower()]
                ben_temp[h]=data_allDict[h.lower()]
            except KeyError:
                print("Key not found: ",h.lower())
                ben_temp[h]=0
        ben_temp= (sorted(ben_temp.items(), key=lambda x: x[1]))[-1]

        ben_counts[ben_temp[0]]=ben_temp[1]
        if ben_temp[1]==0:
            score=0
        else:
            score=(eng_counts[(json_finals[i][j]['eng'])]/ben_temp[1])+0.1


        dict_temp={"eng":(json_finals[i][j]['eng']),"pos_tag":json_finals[i][j]['pos_tag'],"eng_score":eng_counts[(json_finals[i][j]['eng'])],"ben":json_finals[i][j]['spanish'],"hin_roman":ben_temp[0],"ben_score":ben_temp[1],"score":score}
        sent_dict.append(dict_temp)
    Main_sent_dict.append(sent_dict)

    # do not mind keys like "ben", "hin_roman"  as can use the same code below for code-mixed sentence generation.

# Choose CMD here, the final value of 'code_mixed' is the code-mixed generated.

CMD=0.5
i=0
hindi_sentence=responsee_hin.strip()
code_mixed=hindi_sentence
words_num_replace=int(CMD*len(sent_dict))
def custom_sortingg(x):
    return (x['score']==0.0,x['score'])
sorted_sent_dict= sorted(sent_dict, key=custom_sortingg, reverse=True)
#sorted_sent_dict= sorted(sent_dict, key=lambda x: x['score'],reverse=True)

#while(sent_dict)
'''
for k in range(words_num_replace):
    print(k)
    code_mixed=replace_word(code_mixed,sorted_sent_dict[k]['hin'],sorted_sent_dict[k]['eng'])
    print(code_mixed)
'''
initi=words_num_replace
for i in range(len(sorted_sent_dict)):
    if((sorted_sent_dict[i]['score'])==0.0):
        initi=max(0,initi-1)
        code_mixed=replace_word(code_mixed,sorted_sent_dict[i]['hin'],sorted_sent_dict[i]['eng'])
        print(i,initi)
        print(code_mixed)
    else:
        if(initi==0):
            break
        else:
            initi=max(0,initi-1)
            code_mixed=replace_word(code_mixed,sorted_sent_dict[i]['hin'],sorted_sent_dict[i]['eng'])
            print(i,initi)
            print(code_mixed)

#english-bengali
bengali_sentence=responsee_ben.strip()
#bengali_sentence=inflec_indep_sent.strip()
code_mixed=bengali_sentence
words_num_replace=int(CMD*len(sent_dict))
#def custom_sortingg(x):
#    return (x['score']=s=0.0,x['score'])
def custom_sortingg(x):
    return (x['score'] == 0.0, x['eng_score'] if x['score'] == 0.0 else x['score'])
sorted_sent_dict= sorted(sent_dict, key=custom_sortingg, reverse=True)
initi=words_num_replace
for i in range(len(sorted_sent_dict)):
    if((sorted_sent_dict[i]['score'])==0.0):
        initi=max(0,initi-1)
        code_mixed=replace_word(code_mixed,sorted_sent_dict[i]['ben'],sorted_sent_dict[i]['eng'])
        print(i,initi)
        print(code_mixed)
    else:
        if(initi==0):
            break
        else:
            initi=max(0,initi-1)
            code_mixed=replace_word(code_mixed,sorted_sent_dict[i]['ben'],sorted_sent_dict[i]['eng'])
            print(i,initi)
            print(code_mixed)


"""English as Matrix language"""

dof=1
Eng_sentt="The questions are of four types"
english_sentence=Eng_sentt.lower().strip()
code_mixed=english_sentence
words_num_replace=int(dof*len(sent_dict))
def custom_sortingg(x):
    return (x['score']==0.0,x['score'])
sorted_sent_dict= sorted(sent_dict, key=custom_sortingg, reverse=False)
#sorted_sent_dict= sorted(sent_dict, key=lambda x: x['score'],reverse=True)

#while(sent_dict)
'''
for k in range(words_num_replace):
    print(k)
    code_mixed=replace_word(code_mixed,sorted_sent_dict[k]['hin'],sorted_sent_dict[k]['eng'])
    print(code_mixed)
'''
initi=words_num_replace
for i in range(len(sorted_sent_dict)):
        if(initi==0):
            break
        else:
            initi=max(0,initi-1)
            code_mixed=replace_word(code_mixed,sorted_sent_dict[i]['eng'],sorted_sent_dict[i]['hin_roman'])
            print(i,initi)
            print(code_mixed)
print("Final CodeMixed: ",code_mixed )