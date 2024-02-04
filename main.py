# Streamlit Cloud
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

def unpack_citations(incoming):
    staging = ""
    for x in range(len(incoming['text'].answer)):
        stage2 = incoming['text'].answer[x].substring_quote
        staging = f"{staging}  \n\n {x+1}:{stage2}"
    return staging

def unpack_answer(incoming):
    staging = ""
    for x in range(len(incoming['text'].answer)):
        stage2 = incoming['text'].answer[x].fact
        staging = f"{staging} \n\n {x+1}. {stage2}"
    return staging


vectordb = data_load(diagnostic_mode=diagnostic_mode)
st.title("SOP with citations")
intro = st.subheader("Welcome to your SOP guide")

text_input = st.text_input(label="What would you like help with?",value="What happens during turn season? ")


if text_input:
    query = text_input
    context = vectordb.similarity_search(query)
    results = citation_chain(question=query, context=context)
    citations = unpack_citations(results)

    st.subheader("Answer")
    st.markdown(unpack_answer(results))
    col1, col2 = st.columns([.7,.3])
    col1.subheader("Quotations")
    col1.write(citations)

    if 'context' in results and len(results['context']) > 1:
        source1_raw = results['context'][1].metadata['source']
        source1 = cleaner(source1_raw)
        col2.subheader("Sources")
        col2.write(source1[1])

    # st.markdown("[google](www.google.com)") # Example of markdown hyperlink