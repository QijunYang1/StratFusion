

import os
import openai
import time
from openai import OpenAI

from multiprocessing import Pool
from itertools import repeat


def get_sentiment_sample(news, model="gpt-3.5-turbo"):
    system_analysis_prompt =  "You will work as a Sentiment Analysis Expert for Financial News. \
        You will only answer as: \n\n BEARISH, BULLISH, NEUTRAL. No further explanation=."
    messages = []
    messages = [{"role": "system", "content": system_analysis_prompt}]
    messages.append({"role": "user", "content":news})
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def get_sentiment_general(news, model="gpt-3.5-turbo", company=None, output="polar", explanation=False):
    if company:
        company_prompt = f"for {company}" 
    else:
        company_prompt = ""
    if explanation:
        explanation_prompt = f"Your answer will include 2 lines. In the first line, you will answer 1 sentence to analyze why the news is good or bad {company_prompt}."
        explanation_addup = "in the second line, "
    else:
        explanation_prompt = "You will not give explanation to your answer."
        explanation_addup = ""
        
    if output == "polar":
        output_prompt = f"Then, {explanation_addup}you will answer with: BEARISH, BULLISH, NEUTRAL."
    elif output =="score":
        output_prompt = f"Then, {explanation_addup}you will answer with an integer between 1 and 10, with 1 being most BEARISH, 10 being most BULLISH."
    else:
        output_prompt = f"Then, {explanation_addup}you will answer with an integer between 1 and 10, with 1 being most BEARISH, 10 being most BULLISH."
        
    system_analysis_prompt =  f"You will work as a Sentiment Analysis for Financial News {company_prompt}. {explanation_prompt} {output_prompt}"
    messages = []
    messages = [{"role": "system", "content": system_analysis_prompt}]
    messages.append({"role": "user", "content":news})
    client = OpenAI(
        api_key="" # Put your key here!
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    response_message = response.choices[0].message.content
    if explanation:
        ans = response_message.split("\n")
        if len(ans) == 1:
            ans = response_message.split(" ")
            output = ans[-1]
            reason = " ".join(ans[:-1])
        else:
            reason = ans[0]
            output = ans[-1]
        
    else:
        output = response_message
        reason = ""
    
    return output, reason


def get_sentiment_general_parallel(news_list, model="gpt-3.5-turbo", company=None, output="polar", explanation=False, threads=10):
    t0 = time.time()
    with Pool(threads) as p:
        pool_output = p.starmap(get_sentiment_general, zip(news_list, repeat(model), repeat(company), repeat(output), repeat(explanation)))
    t1 = time.time()
    print(f" average running time: {(t1-t0)/len(news_list):.2f} second")
    return pool_output

def get_sentiment_general_sequence(news_list, model="gpt-3.5-turbo", company=None, output="polar", explanation=False, threads=10):
    t0 = time.time()
    pool_output = []
    for i in range(len(news_list)):
        pool_output.append(get_sentiment_general(news_list[i], model, company, output, explanation))
    t1 = time.time()
    print(f" average running time: {(t1-t0)/len(news_list):.2f} second")
    return pool_output