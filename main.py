# Streamlit Cloud deploy for db - sqlite workaround
import os
cwd = os.getcwd()
if cwd[0] != 'C':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

### Import Functions
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_citation_fuzzy_match_chain
from icecream import ic
import re

### Function declarations
def data_load(persist_directory = "db", diagnostic_mode = 0):
    embedding = OpenAIEmbeddings(api_key=st.secrets['OPENAI_API_KEY'])
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    
    if diagnostic_mode == 1:
        if len(vectordb.get()) == 6:
            print("Empty database loaded")
        else: 
            print(len(vectordb.get()))
    
    return vectordb

def citation_chain(question, context, diagnostic_mode = 0):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", api_key=st.secrets['OPENAI_API_KEY'])


    chain = create_citation_fuzzy_match_chain(llm)

    result2 = chain.invoke({'question': question, 'context' : context})
    result_staging = result2['text']

    if diagnostic_mode == 1:
        ic(result_staging)
        #ic(result2)
    return(result2)


def unpack_citations(incoming):
    staging = ""
    for x in range(0,len(incoming['text'].answer)):
        ic(incoming['text'].answer[x].substring_quote)
        stage2 = incoming['text'].answer[x].substring_quote
        stage2 = '  \n\n'.join(stage2)
        stage2 = re.sub("\n\n", "  \n\n", stage2)
        ic(stage2)
        staging = f'{staging}  \n\n <b id="quote{x+1}">{x+1}</b>: {stage2}'
    return staging

def unpack_answer(incoming):
    staging = ""
    for x in range(len(incoming['text'].answer)):
        stage2 = incoming['text'].answer[x].fact
        staging = f'{staging}  \n\n {x+1}. {stage2}[<sup>{x+1}</sup>](#quote{x+1})'
    return staging

def cleaner(inputa : str) -> str:
    regex = r".*\\(.*)"

    match = re.search(regex, inputa) # Access the first group (entire match)
    return match

def abbreviate_titles(source_titles: list) -> list:
    """
    Modifies each title in a list by removing the first 5 and last 4 characters.

    Args:
        source_titles: A list of strings containing the original titles.

    Returns:
        A new list containing the abbreviated titles.
    """

    abbreviated_titles = []
    for source_title in source_titles:
        abbreviated_title = source_title[5:-4]  # Extract the middle portion
        abbreviated_titles.append(abbreviated_title)
    return abbreviated_titles

def ref_search(search_string, results, diagnostic_mode = 0):
    #search_string = "  \n\n".join(search_string)
    ic(search_string)
    for x in range(0, len(results['context'])):
        step_1 = results['context'][x].page_content
        if diagnostic_mode == 1:
            print(f"searching source {x+1}")
            ic(step_1.find(search_string))
        #step_1 = stringify(step_1)
        if diagnostic_mode == 1:
            ic(step_1)
        step_2 = "".join(step_1)
        ic(re.search(f"{step_1}",f"{results['context'][x].page_content}"))
        #if re.search(step_1,results['context'][x].page_content) > 1:
        #    print(f"source for answer {x} found: {results['context'][x].metadata['source']}")
         #   cited = results['context'][x].metadata['source']
          #  if diagnostic_mode == 1:
           #     ic(cited)
            #return cited

def clean_b(input_strings):
    cleaned_strings = ""
    for string in input_strings:
        cleaned = re.sub(r'[^\w\s]', '', string)
    cleaned_strings.append(cleaned)
    
    return cleaned_strings

def cited_rag(query, diagnostic_mode = 0):
    context = vectordb.similarity_search(query)
    with st.spinner(text="Checking the archives"):
        results = citation_chain(question=query, context=context)
    if diagnostic_mode == 1:
        ic(results)
        ic(context)
    citations = unpack_citations(results)
    num_sources = len(results['context'])
    source_titles = []
    source_content = []
    for x in range(0, num_sources):
        source_titles.append(results['context'][x].metadata['source'])
        source_content.append(results['context'][x].page_content)
    ic(results['text'].answer)
    #for answer in results['text'].answer:
     #   ic(ref_search(answer.substring_quote, results=results))

    st.subheader("Answer", anchor="Answer")
    st.markdown(unpack_answer(results), unsafe_allow_html=True)
    col1, col2 = st.columns([.7,.3])
    col1.subheader("Quotations", anchor='Quotations')
    col1.markdown(citations, unsafe_allow_html=True)
    used_chunks = []
    if 'context' in results and num_sources > 0:
        for i in range(0, len(results['text'].answer)):
            search_string = results['text'].answer[i].substring_quote
            if len(search_string) >= 1: search_string = "".join(search_string)
            ic(search_string)
            sources_name = "Search tool failed"
            for x in range(0, num_sources):
                # Need to santize to literal strings somehow
                answer_chunk = re.search(f"{search_string}",f"{results['context'][x].page_content}")
                ic(answer_chunk)
                if answer_chunk != None:
                    sources_name = results['context'][x].metadata['source']
            used_chunks.append(sources_name)


        sources_for_display = ""
        for x in range(0,len(used_chunks)):
            
            sources_for_display = f'{sources_for_display}  \n\n {x+1}. {used_chunks[x]}' # [<sup>{x+1}</sup>](#quote{x+1})

        col2.subheader("Sources", anchor='Sources')
        col2.markdown(sources_for_display)
    
    st.subheader("Sections of sources used")
    source_titles = abbreviate_titles(source_titles)
    tabs = st.tabs(source_titles)
    for label, tab in zip(source_content, tabs):
        with tab:
            st.markdown(label)



### Data Declarations
diagnostic_mode = 0 # turns on checkpoints

vectordb = data_load(diagnostic_mode=diagnostic_mode)
prompta = "I am the operations manager for Bloomington. It is february, what should I be working on?"
promptb = "What documents do I need for an international applicant?"
promptc = "Who is responsible for move out inspections?"
promptd = "I am the on call technician, is the clogged toilet that just got called in an emergency?"
prompte = "What are the steps to gathering a bid?"
promptf = "How should I explain the reconditioning fee?"

### Streamlit declaration
st.set_page_config(layout="wide")
st.title("SOP with citations")
intro = st.subheader("Welcome to your SOP guide")
query = ""
#passphrase = st.text_input(label="Please enter your passcode", value="Speak friend and enter")
code = st.secrets['passcode']
# Create a placeholder for the passcode input
passcode_placeholder = st.empty()

# Check if the user is authenticated
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Display the passcode input if not authenticated
if not st.session_state.authenticated:

    passcode = passcode_placeholder.text_input(label="Please enter your passcode", value="Speak friend and enter", type="password")
    if passcode == st.secrets['passcode']:
        st.session_state.authenticated = True
        passcode_placeholder.empty()  # Clear the passcode input
if st.session_state.authenticated:
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
        cited_rag(query=query, diagnostic_mode=1)
        # st.markdown("[google](www.google.com)") # Example of markdown hyperlink
st.subheader("Interested in using AI to help implement your SOPs?")
st.markdown("Send me a note at ritterstandalpha@gmail.com")

