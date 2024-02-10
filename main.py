# Streamlit Cloud deploy for db - sqlite workaround
import os
cwd = os.getcwd()
if cwd[0] != 'C':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    
# Standard Variables
debug_mode = 0 # Test this file locally
diagnostic_mode = 0 # turns on checkpoints

##############################
# Database Functions
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

def data_load(persist_directory = "db", diagnostic_mode = 0):
    embedding = OpenAIEmbeddings(api_key=st.secrets['OPENAI_API_KEY'])
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    
    if diagnostic_mode == 1:
        if len(vectordb.get()) == 6:
            print("Empty database loaded")
        else: 
            print(len(vectordb.get()))
    
    return vectordb

##############################
# LLM Functions
from icecream import ic
from langchain_openai import ChatOpenAI
from langchain.chains import create_citation_fuzzy_match_chain
import streamlit as st
st.set_page_config(layout="wide")

def highlight(text, span):
    return (
        "..."
        + text[span[0] - 20 : span[0]]
        + "*"
        + "\033[91m"
        + text[span[0] : span[1]]
        + "\033[0m"
        + "*"
        + text[span[1] : span[1] + 20]
        + "..."
    )

def citation_chain(question, context, diagnostic_mode = 0):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", api_key=st.secrets['OPENAI_API_KEY'])


    chain = create_citation_fuzzy_match_chain(llm)
    if diagnostic_mode == 2:
        result = chain.run(question=question, context=context)
        ic(result)

        for fact in result.answer:
            print("Statement:", fact.fact)
            for span in fact.get_spans(context):
                print("Citation:", highlight(context, span))
            print()

    result2 = chain.invoke({'question': question, 'context' : context})
    result_staging = result2['text']

    if diagnostic_mode == 1:
        ic(result_staging)
        #ic(result2)
    return(result2)

if debug_mode == 1:
    print("debug run")
    query = "What happens during turn season?"
    vectordb = data_load(diagnostic_mode=diagnostic_mode)
    docs = vectordb.similarity_search(query)
    citation_chain(question=query, context=docs, diagnostic_mode=diagnostic_mode)

########################################
# UI functions
import re

def cleaner(inputa : str) -> str:
    regex = r".*\\(.*)"

    match = re.search(regex, inputa) # Access the first group (entire match)
    return match

def ref_search(inp_str: str, search_str: str, context_len: int = 20) -> str:
    """
    Search for search_str in inp_str 
    and return substring with context_len 
    characters before and after search_str.
    """
    index = inp_str.find(search_str)
    
    # Handle case where search_str not found
    if index == -1:  
        return "" 
    
    start = max(0, index - context_len)
    end = index + len(search_str) + context_len
    output = inp_str[start:end]
    
    return output

def source_checker(context):
    print(context)

def unpack_citations(incoming):
    staging = ""
    for x in range(0,len(incoming['text'].answer)):
        stage2 = str(incoming['text'].answer[x].substring_quote)
        stage2 = re.sub("\n\n", "  \n\n", stage2)
        staging = f'{staging}  \n\n <b id="quote{x+1}">{x+1}</b>: {stage2}'
    return staging

def unpack_answer(incoming):
    staging = ""
    for x in range(len(incoming['text'].answer)):
        stage2 = incoming['text'].answer[x].fact
        staging = f'{staging}  \n\n {x+1}. {stage2}[<sup>{x+1}</sup>](#quote{x+1})'
    return staging

def cited_rag(query):
    context = vectordb.similarity_search(query)
    with st.spinner(text="Checking the archives"):
        results = citation_chain(question=query, context=context)
    citations = unpack_citations(results)
    num_sources = len(results['context'])
    st.subheader("Answer", anchor="Answer")
    st.markdown(unpack_answer(results), unsafe_allow_html=True)
    col1, col2 = st.columns([.7,.3])
    col1.subheader("Quotations", anchor='Quotations')
    col1.markdown(citations, unsafe_allow_html=True)

    if 'context' in results and num_sources > 0:
        sources = ""
        sources_b = []
        count = 1
        used_chunks = []
        for x in range(0,num_sources):
            #ic(results)
            num_cit = len(results['text'].answer)
            source_raw = results['context'][x].metadata['source']
            ic(source_raw)
            full_citation = results['context'][x].page_content
            #print(dir(full_citation))
            used_chunks.append(full_citation)
            #ic(full_citation)
            source_staging = cleaner(source_raw)[1]
            ic(source_staging)
            sources_b.append(source_staging)
            if sources.find(source_staging) == -1:
                #ic(sources.find("source_staging"))
                sources = f'{sources} {count}.{source_staging} \n\n'
                count += count
                
        col2.subheader("Sources", anchor='Sources')
        col2.write(sources)


vectordb = data_load(diagnostic_mode=diagnostic_mode)
prompta = "I am the operations manager for Bloomington. It is february, what should I be working on?"
promptb = "What documents do I need for an international applicant?"
promptc = "Who is responsible for move out inspections?"
promptd = "I am the on call technician, is the clogged toilet that just got called in an emergency?"
prompte = "What are the steps to gathering a bid?"
promptf = "How should I explain the reconditioning fee?"


st.title("SOP with citations")
intro = st.subheader("Welcome to your SOP guide")
query = ""
#passphrase = st.text_input(label="Please enter your passcode", value="Speak friend and enter")
code = st.secrets['passcode']
if st.text_input(label="Please enter your passcode", value="Speak friend and enter") == code:
    st.markdown("<span style='display: grid; place-items: center;'>Not sure where to start? Here are some of my favorite prompts, it takes about 6-8 seconds to answer right now</span>", unsafe_allow_html=True)
    cola, colb, colc, cold, cole, colf = st.columns(6)
    with cola:
        if st.button(prompta, key="prompta"):
            query = prompta
    with colb:
        if st.button(promptb, key="promptb"): 
            query = promptb
    with colc:
        if st.button(promptc, key="promptc"): 
            query = promptc  
    with cold:
        if st.button(promptd, key="promptd"): 
            query = promptd
    with cole:
        if st.button(prompte, key="prompte"): 
            query = prompte
    with colf:
        if st.button(promptf, key="promptf"): 
            query = promptf
    

    user_question = st.text_input(label="What would you like help with?",placeholder="What happens during turn season? ", key="user_question")
    if user_question:
        query = user_question

    if query:
        cited_rag(query=query)
        # st.markdown("[google](www.google.com)") # Example of markdown hyperlink

else: 
    st.subheader("Interested in using AI to help implement your SOPs?")
    st.markdown("Send me a note at ritterstandalpha@gmail.com")
