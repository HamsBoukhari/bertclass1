import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForSequenceClassification
import torch
import re
import xml.etree.ElementTree as ET
def clean_sequence(input_seq):
    pattern = r'[<\"/]'
    cleaned_seq1 = re.sub(pattern, '', input_seq)
    cleaned_seq = cleaned_seq1.replace('>',' ')
    return cleaned_seq
def predict(seq,model):
    model.eval()
    input_ids = tokenizer.encode(seq, return_tensors='pt')
    output = model.generate(input_ids, max_length=1000, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    output_sequence = decoded_output[len(seq):].strip()
    return output_sequence
def predict1(seq):
    model.eval()
    encoding = tokenizer1(seq, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
    if preds.item() == 0:
        return "Full Service"
    elif preds.item() == 1:
        return "Mark As Give Up"
    elif preds.item() == 2:
        return "Give Up"
    elif preds.item() == 3:
        return "Split On Two Accounts"
    elif preds.item() == 4:
        return "Split On Three Accounts"
    else:
        return "Split On Four Accounts"
def extract_giveup(xml_string):
    root = ET.fromstring(xml_string)
    hdr = root.find('Hdr')
    if hdr is not None:
        root.remove(hdr)
    instrmt = root.find('Instrmt')
    if instrmt is not None:
        root.remove(instrmt)
    pty_elements = root.findall('Pty')
    for i in range(3):
        if i < len(pty_elements):
            root.remove(pty_elements[i])
    return ET.tostring(root, encoding='utf-8', method='xml').decode()
def extract_first_alloc(xml_string):
    root = ET.fromstring(xml_string)
    first_alloc = root.find(".//Alloc")
    return ET.tostring(first_alloc, encoding='utf-8', method='xml').decode()
def extract_second_alloc(xml_string):
    root = ET.fromstring(xml_string)
    all_allocs = root.findall(".//Alloc")
    second_alloc = all_allocs[1]
    return ET.tostring(second_alloc, encoding='utf-8', method='xml').decode()
def extract_third_alloc(xml_string):
    root = ET.fromstring(xml_string)
    all_allocs = root.findall(".//Alloc")
    third_alloc = all_allocs[2]
    return ET.tostring(third_alloc, encoding='utf-8', method='xml').decode()
def extract_fourth_alloc(xml_string):
    root = ET.fromstring(xml_string)
    all_allocs = root.findall(".//Alloc")
    fourth_alloc = all_allocs[3]
    return ET.tostring(fourth_alloc, encoding='utf-8', method='xml').decode()
@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model_fullser = GPT2LMHeadModel.from_pretrained("hams2/fullservice")
    model_mark1 = GPT2LMHeadModel.from_pretrained("hams2/markasgup1")
    model_mark2 = GPT2LMHeadModel.from_pretrained("hams2/markasgup2")
    model_giveup = GPT2LMHeadModel.from_pretrained("hams2/giveup")
    model_split1 = GPT2LMHeadModel.from_pretrained("hams2/split1")
    model_split2 = GPT2LMHeadModel.from_pretrained("hams2/split2")
    tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained("hams2/bertclass")
    return tokenizer,model_fullser,model_mark1,model_mark2,model_giveup,model_split1,model_split2,tokenizer1,model
tokenizer,model_fullser,model_mark1,model_mark2,model_giveup,model_split1,model_split2,tokenizer1,model = get_model()
trade_msg = st.text_area('Trade Message')
sgw_op = st.text_area('SGW Operation')
button = st.button("Predict")
if (trade_msg and sgw_op) and button:
    trade_msg1 = trade_msg.replace('\n','')
    sgw_op1 = sgw_op.replace('\n','')
    sgw_op2 = clean_sequence(sgw_op1)
    workflow = predict1(sgw_op2)
    if workflow=="Full Service":
        st.write("Workflow: ",workflow)
        input_seq = trade_msg1+' '+sgw_op1
        seq = clean_sequence(input_seq)  
        ccp_mssg = predict(seq,model_fullser)
        st.write("CCP Message: ",ccp_mssg)
    elif workflow == "Mark As Give Up":
        st.write("Workflow: ",workflow)
        input_seq = trade_msg1+' '+sgw_op1
        seq = clean_sequence(input_seq)  
        ccp_mssg1 = predict(seq,model_mark1)
        st.write("CCP Message 1: ",ccp_mssg1)        
        ccp_mssg2 = predict(seq,model_mark2)
        st.write("CCP Message 2: ",ccp_mssg2) 
    elif workflow == "Give Up":
        st.write("Workflow: ",workflow)
        ex_sgw_op1 = extract_giveup(sgw_op1)
        input_seq = trade_msg1+' '+ex_sgw_op1
        seq = clean_sequence(input_seq)
        ccp_mssg = predict(seq,model_giveup)
        st.write("CCP Message: ",ccp_mssg) 
    elif workflow == "Split On Two Accounts":
        st.write("Workflow: ",workflow)
        ex_sgw_op1 = extract_first_alloc(sgw_op1)
        ex_sgw_op2 = extract_second_alloc(sgw_op1)
        input_seq1 = trade_msg1+' '+ex_sgw_op1
        input_seq2 = trade_msg1+' '+ex_sgw_op2
        seq1 = clean_sequence(input_seq1)
        seq2 = clean_sequence(input_seq2)
        ccp_mssg1 = predict(seq1,model_split1)
        st.write("CCP Message 1: ",ccp_mssg1)        
        ccp_mssg2 = predict(seq2,model_split2)
        st.write("CCP Message 2: ",ccp_mssg2)
    elif workflow == "Split On Three Accounts":
        st.write("Workflow: ",workflow)
        ex_sgw_op1 = extract_first_alloc(sgw_op1)
        ex_sgw_op2 = extract_second_alloc(sgw_op1)
        ex_sgw_op3 = extract_third_alloc(sgw_op1)
        input_seq1 = trade_msg1+' '+ex_sgw_op1
        input_seq2 = trade_msg1+' '+ex_sgw_op2
        input_seq3 = trade_msg1+' '+ex_sgw_op3
        seq1 = clean_sequence(input_seq1)
        seq2 = clean_sequence(input_seq2)
        seq3 = clean_sequence(input_seq3)
        ccp_mssg1 = predict(seq1,model_split1)
        st.write("CCP Message 1: ",ccp_mssg1)        
        ccp_mssg2 = predict(seq2,model_split2)
        st.write("CCP Message 2: ",ccp_mssg2)                 
        ccp_mssg3 = predict(seq3,model_split2)
        st.write("CCP Message 3: ",ccp_mssg3)    
    elif workflow == "Split On Four Accounts":
        st.write("Workflow: ",workflow)
        ex_sgw_op1 = extract_first_alloc(sgw_op1)
        ex_sgw_op2 = extract_second_alloc(sgw_op1)
        ex_sgw_op3 = extract_third_alloc(sgw_op1)
        ex_sgw_op4 = extract_fourth_alloc(sgw_op1)
        input_seq1 = trade_msg1+' '+ex_sgw_op1
        input_seq2 = trade_msg1+' '+ex_sgw_op2
        input_seq3 = trade_msg1+' '+ex_sgw_op3
        input_seq4 = trade_msg1+' '+ex_sgw_op4
        seq1 = clean_sequence(input_seq1)
        seq2 = clean_sequence(input_seq2)
        seq3 = clean_sequence(input_seq3)
        seq4 = clean_sequence(input_seq4)
        ccp_mssg1 = predict(seq1,model_split1)
        st.write("CCP Message 1: ",ccp_mssg1)        
        ccp_mssg2 = predict(seq2,model_split2)
        st.write("CCP Message 2: ",ccp_mssg2)                 
        ccp_mssg3 = predict(seq3,model_split2)
        st.write("CCP Message 3: ",ccp_mssg3)  
        ccp_mssg4 = predict(seq4,model_split2)
        st.write("CCP Message 4: ",ccp_mssg4)  
        