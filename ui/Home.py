import streamlit as st
import streamlit.components.v1 as components
import os
import re
import json
import requests
import pandas as pd
import numpy as np
import time
import configparser
# Import utility functions
import sys

config = configparser.ConfigParser()
config.read('config.ini')
st.set_page_config(layout="wide")

if "num_combos" not in st.session_state:
    st.session_state.num_combos = 0
if "question" not in st.session_state:
    st.session_state.question = ''
if "results" not in st.session_state:
    st.session_state.results = []
if "clear" not in st.session_state:
    st.session_state.clear = 0
if "refresh" not in st.session_state:
    st.session_state.refresh = 0
if "combo_names" not in st.session_state:
    st.session_state.combo_names = {}
if "entered_ques" not in st.session_state:
    st.session_state.entered_ques = ""
if "submit_ques" not in st.session_state:
    st.session_state.submit_ques = False

def print_labels_types(labels, types):
    labels_df = pd.DataFrame(list(zip(labels, types)), columns=["Predicted Label", "Types"])
    st.markdown("###### Predicted Labels") 
    st.dataframe(labels_df, hide_index=True)

def make_clickable(link, strip=True):
    link, text = link.split(':::')
    if strip:
        link = link[1:-1]
    return f'<a target="_blank" href="{link}">{text}</a>'

@st.cache_data(show_spinner = False)
def add_result(question, model_name, embedding, _config, _environment):
    url = _config[_environment]['entitylinker'] + model_name+ '/' + embedding
    headers = {"Content-Type": "application/json"}
    responses = []
    data = {"question": question}
    resp = requests.post(url, headers=headers, data=json.dumps(data))
    print(resp)
    resp_json = resp.json()
    result = {"question": question, "model_name": model_name, "embed_name": embedding, "responses":resp_json}
    return result

def update_results(result):
    st.session_state.results.append(result)
    st.session_state.num_combos += 1

def types_to_str(typs):
    typs_str = ''
    for t in typs:
        t = "https://dbpedia.org/ontology/"+t+":::"+t
        typs_str = typs_str + make_clickable(t, False) + " "
    return typs_str

def display_table(res):
    #df_final = pd.DataFrame(columns=("Label", "Types", "Distance", "Entity"))
    df_final = pd.DataFrame(columns=("Label", "Types",  "Entity"))
    for idx,entitylinkingresult in enumerate(res['responses']['entitylinkingresults']):
        label = entitylinkingresult['label']
        typs = entitylinkingresult['type']
        resp_json = res['responses']['entitylinkingresults'][idx]['result']
        print(resp_json)
        dist = round(resp_json[0][0], 2)
        link = resp_json[0][1][0]
        ent_label = resp_json[0][1][1]
        link_str = link + ":::" + ent_label
        types_str = types_to_str(typs)
        #obj = {"Label": ent_label, "Types": types_str, "Distance": dist, "Entity": link_str}
        obj = {"Label": ent_label, "Types": types_str, "Entity": link_str}
        #df = pd.DataFrame(obj, columns=("Label", "Types", "Distance", "Entity"), index=[0])
        df = pd.DataFrame(obj, columns=("Label", "Types", "Entity"), index=[0])
        df_final = pd.concat([df_final, df], ignore_index=True)
    df_final['Entity'] = df_final['Entity'].apply(make_clickable)
    st.write(df_final.to_html(escape = False, index=False, justify='center'), unsafe_allow_html = True)

def display_res(res):
    for idx,entitylinkingresult in enumerate(res['responses']['entitylinkingresults']):
        label = entitylinkingresult['label']
        resp_json = res["responses"]['entitylinkingresults'][idx]['result']
        dists = [item[0] for item in resp_json]
        links = [item[1][0] for item in resp_json]
        ent_labels = [item[1][1] for item in resp_json]
        st.markdown("**Label " + str(idx) + "**: " + label)
        link_strs = []
        for i in range(len(links)):
            link_str = links[i] + ":::" + ent_labels[i]
            link_strs.append(link_str)
        #df = pd.DataFrame((list(zip(dists, link_strs))), columns=("Distance", "Entity"))
        df = pd.DataFrame((list( link_strs)), columns=( "Entity",))
        df['Entity'] = df['Entity'].apply(make_clickable)
        st.write(df.to_html(escape = False), unsafe_allow_html = True)
        st.divider()
        idx = idx + 1

def refresh():
    st.session_state.refresh = 1

def clear_callback():
    st.session_state.results = []
    st.session_state.num_combos = 0
    st.session_state.combo_names = {} 
    refresh()

def del_combo(del_model):
    idx = st.session_state.combo_names[del_model]
    st.session_state.results.pop(idx)
    st.session_state.num_combos -= 1
    st.session_state.combo_names.pop(del_model)
    st.session_state.refresh = 1

def use_entered_ques():
    st.session_state.selection_1 = ""

def use_selected_ques():
    st.session_state.entered_ques = st.session_state.selection_1
    st.session_state.selection_1 = ""

def main(config, environment='staging'):
    with st.sidebar:
        st.header('DBLP Entity Linker')

    # Custom style
    st.markdown("""
    <style type="text/css">
    div[data-testid="stHorizontalBlock"]{
    border:10px;
    padding:5px;
    border-radius:10px;
    background:#f9f9f9;
    }
    div[class="block-container css-z5fcl4 e1g8pov64"]{
    padding:3rem;
    }
    table {
    word-wrap: break-all; 
    flex:1;
    }
    td {
    text-align: left;
    word-break:break-word;
    }
    th {
    
    }
    div[class="st-c7 st-cl st-cm st-ae st-af st-ag st-ah st-ai st-aj st-cn st-co"]{
    font-size:smaller;
    }
    .Aligner {
    display: flex;
    align-items: center;
    justify-content: center;
    }
    .flex-container {
    display: flex;
    flex-direction: row;
    justify-content: space-between;         
    }
    div[data-testid="stVerticalBlockNew"]{
    border-style: solid;
    border-width: 1px;
    border-color: #ff4b4b;
    border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Mapping names
    model_name_display = {"T5-small":"t5-small", "T5-base":"t5-base"}
    model_name_display_inverse = {v: k for k, v in model_name_display.items()}
    embed_name_display = {"TransE":"transe", "ComplEx":"complex", "DistMult":"distmult"}
    model_name = ''
    embedding = ''
    with st.form("main-form"):
        # Question
        sample_ques = st.selectbox('Select from samples', ["","Who were the co-authors of Ashish Vaswani in the paper ‘Attention is all you need’?", "When was the paper ‘An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale’ published?", "Which papers were written by Yann LeCun and Yoshua Bengio?","When was Adam introduced for stochastic optimization?"], index=0, key="selection_1")
        entered_ques = st.text_input('Enter question',  key="entered_ques")
        col1,col2 = st.columns([0.2,0.2])
        with col1:
            st.markdown('###### 1. Label Predictor Model')
            model_name = st.radio('Select Model', options=model_name_display.keys(), label_visibility = 'collapsed', horizontal=False)
        with col2:
            st.markdown('###### 2. Entity Ranker Embeddings')
            embedding = st.radio('Select Embeddings',options=embed_name_display.keys(), label_visibility='collapsed', horizontal=False)
        st.write("Tip: For sample question to be answered, the 'Enter question' field must be empty.")
        submit_btn = st.form_submit_button("Submit")
    
    col1_out, col2_out = st.columns([0.5, 0.5])
    
    # Add the result if submit pressed
    if submit_btn:
        ques = ''
        if len(entered_ques):
            ques = entered_ques
        else:
            ques = sample_ques
        if len(ques):
            question = ques
            st.session_state.question = question

            if model_name == '' and embedding == '':
                model_name = 'T5-small'
                embedding = 'TransE'

            combo_name = question + " + " + model_name + " + " + embedding
            if combo_name in st.session_state.combo_names:
                st.error("This combination already exists. Please try another.")
            else:
                st.session_state.combo_names[combo_name] = st.session_state.num_combos
                with col1_out:
                    with st.spinner('Linking...'):
                        result = add_result(question, model_name_display[model_name], embed_name_display[embedding], config, environment)
                        update_results(result)
        else:
            st.error('Please enter or select a question')
        
    # Print the results in two columns
    for i in range(st.session_state.num_combos):
        result = list(st.session_state.results)[st.session_state.num_combos - 1 - i]
        combo_heading = result["model_name"] + " + " + result["embed_name"]
        if i % 2 == 0:
            with col1_out:
                #st.markdown('<div class"stVerticalBlockNew">', unsafe_allow_html=True)
                with st.expander("Question", expanded=True):
                    st.markdown(result["question"])
                with st.expander(combo_heading, expanded=True):
                    display_table(st.session_state.results[st.session_state.num_combos - 1 - i])
                with st.expander('Ranked Entities', expanded=False):
                    display_res(st.session_state.results[st.session_state.num_combos - 1 - i])
                st.divider()
                #st.markdown('</div>', unsafe_allow_html=True)
        else:
            with col2_out:
                #st.markdown('<div class="stVerticalBlockNew">', unsafe_allow_html=True)
                with st.expander("Question", expanded=True):
                    st.markdown(result["question"])
                with st.expander(combo_heading, expanded=True):
                    display_table(st.session_state.results[st.session_state.num_combos - 1 - i])
                with st.expander('Ranked Entities', expanded=False):
                    display_res(st.session_state.results[st.session_state.num_combos - 1 - i])
                st.divider()
                #st.markdown('</div>', unsafe_allow_html=True)
    
    # Display buttons
    del_btn = False
    if st.session_state.num_combos != 0:
        delete_form = st.sidebar.form('Form2')
        with delete_form:
            st.write("Clear workspace")
            del_list = ['Select combination']
            del_list.extend(reversed(list(st.session_state.combo_names.keys())))
            del_list.append('All')
            del_model = st.selectbox('', options=del_list, label_visibility='collapsed', index=0)
            if st.session_state.num_combos == 0:
                disable_delete = True
            else:
                disable_delete = False
            del_btn = st.form_submit_button('Remove', help="Remove a result from the ones displayed.", type='primary', disabled=disable_delete)

    # Delete combination
    if del_btn:
        if del_model == 'Select combination':
            with st.sidebar:
                st.error("Please select a combination")
        elif del_model == 'All':
            clear_callback()
        else:
            del_combo(del_model)
    
#    if st.session_state.refresh==1:
#        st.session_state.refresh = 0
#        time.sleep(3)
#        st.experimental_rerun()     
#    print(st.session_state)

if __name__ == '__main__':
    if sys.argv[1] == 'production':
        main(config, 'production')
    else:
        main(config, 'staging')
