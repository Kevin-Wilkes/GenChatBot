from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import re
import spacy
import ast
import re
import dash
import numpy as np
from typing import Dict, List
from dash import html, dcc, Input, Output, State,Dash
from dash.exceptions import PreventUpdate
import dash_cytoscape as cyto
import threading, webbrowser

nlp = spacy.load("en_core_web_sm")
rel_re = re.compile(r"your\s+(\w+)\s+name+\s+is\s+(\w+)", re.IGNORECASE)

PERSON_NOUNS = []
with open('./Filter_data/Person_Nouns.txt', 'r') as file:
    PERSON_NOUNS = list(ast.literal_eval(file.read()))

FAMILY_RELATIONS = []
with open('./Filter_data/Family_Relations.txt', 'r') as file:
    FAMILY_RELATIONS = list(ast.literal_eval(file.read()))

PLURAL_NOUN = []
with open('./Filter_data/Plural_Nouns.txt', 'r') as file:
    PLURAL_NOUN = list(ast.literal_eval(file.read()))

INVERSE_RELATIONS = {}
with open('./Filter_data/Inverse_Relations.txt', 'r') as file:
    INVERSE_RELATIONS = ast.literal_eval(file.read())

GENDER_CONFIRMATION = {}
with open('./Filter_data/Gender_Confirmation.txt', 'r') as file:
    GENDER_CONFIRMATION = ast.literal_eval(file.read())

conversation_history = {}
with open('./Bot_Data/Conversation_History.txt', 'r') as file:
    conversation_history = ast.literal_eval(file.read())

model_name = "deepseek-ai/deepseek-llm-7b-chat"

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

    
#model = PeftModel.from_pretrained(base_model, "./Finetunned_Deepseek_Model")

def relation_list_parser(relation):
    relation_list = [i for i in relation.split('\n') if len(i) > 0]
    return relation_list

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
            pad_token_id=tokenizer.pad_token_id,
            temperature=1.0,
            top_p=0.85,
            do_sample=True
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    
    clean_response = response[response.rfind('<|im_start|>assistant\n')+22:len(response)-10]
    
    conversation_history[bot_name] = conversation_history[bot_name] + "<|im_start|>\n" + clean_response + "\n<|im_end|>\n"  # Store assistant response

    branch_Word = branchable(clean_response)
    if len(branch_Word) > 0:
        relation_handler(bot_name,branch_Word)

    with open("./Bot_Data/Conversation_History.txt", 'w') as file:
        file.write(json.dumps(conversation_history))
    
    return clean_response

def branchable(response):
    global PERSON_NOUNS
    global FAMILY_RELATIONS
    doc = nlp(response)
    references = []

    for token in doc:
        if token.text in FAMILY_RELATIONS or token.text[:-1] in FAMILY_RELATIONS:
            phrase = f"{token.text}"
            references.append(phrase.lower())
      
    for token in doc:
        if token.dep_ == "poss" and token.head.pos_ in {"NOUN", "PROPN"}:
            noun = token.head.lemma_.lower()
            if noun in PERSON_NOUNS:
                phrase = f"{token.head.text}"
                references.append(phrase.lower())

    
    return list(set(references))


#relation is either a name or relationship like "your doctor"
def relation_handler(bot_name, relations):
    global PLURAL_NOUN
    global PERSON_NOUNS
    generated_relations = conversation_history[bot_name].split('~~')[1]
    print(relations)
    #print(generated_relations)
    for relation in relations:
        if relation not in generated_relations:
            trimmed = relation.lower().removeprefix("my ").strip()
            doc = nlp(trimmed)
            noun = next((t for t in doc if t.pos_ == "NOUN"), None)
            
            lemma = noun.lemma_ if noun else trimmed
            is_plural = (noun and noun.tag_ in ("NNS", "NNPS")) or (lemma in PLURAL_NOUN)
            
            if is_plural:
                summary_response = f'''{conversation_history[bot_name]}
                <|im_start|>user
                list around three names for your {trimmed}. Write only the name and their relation to you. Do not add any explanation or commentary.
                Respond in english.
                <|im_end|>
                <|im_start|>assistant
                '''
            else:
                summary_response = f'''{conversation_history[bot_name]}
                <|im_start|>user
                What is the name for your {trimmed}. Just say the name and nothing else. Respond in english.
                <|im_end|>
                <|im_start|>assistant
                '''
    
            inputs = tokenizer(summary_response, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=.1,
                    top_p=0.9,
                    do_sample=True
                )
            response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
            
            clean_response = response[response.rfind('<|im_start|>assistant\n')+22:len(response)-10]
            #print(clean_response)
    
            clean_response = clean_response.strip().split('\n')
            context_injection = ""
            if is_plural:
                
                context_injection = "your " + relation + " names are the following:\n"
                change = False
                name = ""
                for i in clean_response:
                    
                    doc = nlp(i)
                    
                    for ent in doc.ents:
                        if ent.label_ == "PERSON":
                            name = ent.text.split()[0]
                            break
                    #print(name, len(name))
                    
                    name_relation_parse = ""
                    for x in i.split():
                        if x.lower() in PERSON_NOUNS:
                            name_relation_parse = x

                     #name and relation were parsed
                    if len(name) > 0 and len(name_relation_parse) > 0:

                        #relation doesn't already exist and no duplicate names (confuses the model)
                        if (name_relation_parse not in generated_relations) and (name not in generated_relations) and name not in conversation_history.keys():
                            
                            context_injection += "Your "+ name_relation_parse + " name is "+ name + "\n"
                            change = True

                            bot_creation(bot_name, name, name_relation_parse, generated_relations + context_injection)
                            
                if not change:
                    context_injection = "" #to avoid injecting unformatted mess
            else:
                doc = nlp(clean_response[0])
                name = next((ent.text for ent in doc.ents if ent.label_ == "PERSON"), "")
                if len(name) > 0 and name not in conversation_history.keys():
                    context_injection = "your " + relation + " 's name is " + name+".\n"

                    bot_creation(bot_name, name, relation, generated_relations + context_injection)
            
            generated_relations += context_injection #update generated_relations so no duplication happens during future iterations

            #print(clean_response)

            


    #update context of the bot for better responses in the future
    before = conversation_history[bot_name].split('~~')[0]
    after = conversation_history[bot_name].split('~~')[2]
    conversation_history[bot_name] = before +"~~"+generated_relations+"~~"+ after 

    #print(generated_relations)

def bot_creation(bot_creator, new_bot, relation_to_creator, creator_relations):
    global FAMILY_RELATIONS
    global INVERSE_RELATIONS

    gender = conversation_history[bot_creator].split('~~')[0][-1]
    if gender == "M":
        gender = 0
    else:
        gender = 1
        
    #print(new_bot,relation_to_creator,creator_relations)
    summary_querry = f'''
    <|im_start|>user
    Make a 3 to 4 sentence summary about {new_bot}. Were {new_bot} is {bot_creator}'s {relation_to_creator}. Include information of their work life, family, and social life if they have them. 
    Don't talk about it being a summary.
    <|im_end|>
    <|im_start|>assistant
    '''
    
    inputs = tokenizer(summary_querry, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            temperature=.8,
            top_p=0.8,
            do_sample=True
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    context_guidance = '''Don't talk like you are AI or a chatbot.
    You aren't an assistant. You enjoy small talk. You are here to have fun and relax. avoid using the word assist.
    Here names and your relations you have with people:
    '''
    new_bot_context = "<|im_start|>system\n Your name is " + new_bot + ". " +response[response.rfind('<|im_start|>assistant\n')+22:len(response)-10]+context_guidance

    base_relation = ""
    #print("\n\n"+new_bot_context+"\n\n")
        
    if relation_to_creator in FAMILY_RELATIONS:
        relations = relation_list_parser(creator_relations)
        
        if len(INVERSE_RELATIONS[relation_to_creator]) > 1:
            base_relation = f"Your {INVERSE_RELATIONS[relation_to_creator][gender]} name is {bot_creator}\n"
        else:
            base_relation = f"Your {INVERSE_RELATIONS[relation_to_creator][0]} name is {bot_creator}\n"

        bot_gender_confirmation = GENDER_CONFIRMATION[relation_to_creator.lower()]
                
    else:
        base_relation = f"Your {relation_to_creator} name is {bot_creator}"
        bot_gender_confirmation = "M" if np.random.rand() > 0.5 else "F"
        
    #print(base_relation)
    new_bot_context += bot_gender_confirmation + "~~" + base_relation + "~~\n<|im_end|>"

    conversation_history.update({new_bot:new_bot_context})

app = Dash(__name__, suppress_callback_exceptions=True)
submission_height = "40px"

# --- Layout ---
app.layout = html.Div(
    [
        # Sidebar (tabs + dynamic Bot List)
        html.Div(
            [
                dcc.Tabs(
                    id="main-tabs", value="tab-chat", vertical=True,
                    style={"backgroundColor":"#131313","padding":"10px","boxSizing":"border-box"},
                    children=[
                        dcc.Tab(label="Chat",          value="tab-chat", style={"backgroundColor":"#131313","color":"#888","padding":"12px"}, selected_style={"backgroundColor":"#1e1e1e","color":"#fff","fontWeight":"bold"}),
                        dcc.Tab(label="Visualization", value="tab-viz",   style={"backgroundColor":"#131313","color":"#888","padding":"12px"}, selected_style={"backgroundColor":"#1e1e1e","color":"#fff","fontWeight":"bold"})
                    ]
                ),
                html.Div("Bot List", style={"color":"#fff","fontWeight":"bold","margin":"20px 0 8px 0"}),
                dcc.Tabs(
                    id="bot-selector",
                    vertical=True,
                        style={
                            "backgroundColor": "#131313",
                            "display": "flex",
                            "flexDirection": "column",
                            "overflowY": "auto",
                            "height": "calc(100vh - 120px)"  
                        }
                    # children filled by callback
                )
            ],
            style={"width":"200px","height":"100vh","overflowY":"auto","backgroundColor":"#131313","padding":"10px","boxSizing":"border-box","boxShadow":"2px 2px 5px rgba(0,0,0,0.7)"}
        ),

        # Main content area
        html.Div(
            id="tab-content",
            style={"flex":1,"display":"flex","flexDirection":"column","height":"100vh","overflow":"hidden","padding":"20px","boxSizing":"border-box","backgroundColor":"#2e2e2e"}
        ),

        # Store for chat histories
        dcc.Store(id="chat-histories"),

        # Interval to poll your global conversation_history every 2s
        dcc.Interval(id="refresh-interval", interval=2000, n_intervals=0)
    ],
    style={"display":"flex","height":"100vh","overflow":"hidden","margin":0,"padding":0}
)

#viz layout
viz_layout = html.Div(
    [
        # 1) the graph itself takes all remaining room
        cyto.Cytoscape(
            id="bot-graph",
            elements=[],              # populated by your update_graph callback
            layout={"name": "cose"},
            style={
                "width": "100%",
                "flex": "1 1 auto",   # magic flex to fill the right pane
                "boxSizing": "border-box"
            },
            stylesheet=[
                {
                    "selector": "node",
                    "style": {
                        "label": "data(label)",
                        "background-color": "#0074D9",
                        "color": "#fff",
                        "text-valign": "center",
                        "text-halign": "center",
                        "font-size": "12px"
                    }
                },
                {
                    "selector": "edge",
                    "style": {
                        "curve-style": "bezier",
                        "target-arrow-shape": "triangle",
                        "line-color": "#ccc",
                        "target-arrow-color": "#ccc",
                        "label": "data(relation)",
                        "font-size": "10px",
                        "text-rotation": "autorotate",
                        "text-margin-y": -10
                    }
                }
            ]
        ),

        # 2) info panel pinned immediately below
        html.Div(
            children="Click a node to see its last messages.",
            id="node-info",
            style={
                "flex": "0 0 auto",      # only as tall as its content
                "padding": "10px",
                "color": "#fff",
                "whiteSpace": "pre-wrap",
                "backgroundColor": "#2e2e2e",
                "borderRadius": "6px",
                "marginTop": "10px",
                "maxHeight": "25%",       # prevent it from growing too tall
                "overflowY": "auto"
            }
        ),
        html.Div(id="graph-error", style={
            "flex": 1,
            "display": "flex",
            "flexDirection": "column",
            "height": "100vh",
            "overflow": "hidden",   # ← this is clipping your content
            "padding": "20px",
            "boxSizing": "border-box",
            "backgroundColor": "#2e2e2e"
        })
    ],
    style={
        "flex": 1,
        "display": "flex",
        "flexDirection": "column",
        "height": "100%",    # fill the tab-content container
        "boxSizing": "border-box"
    }
)

# 2) Callback to build nodes+edges from your dash store + relations dict
@app.callback(
    Output("bot-graph",   "elements"),
    Output("graph-error", "children"),
    Input("main-tabs",       "value"),
    State("chat-histories",  "data"),
)
def update_graph(tab, store_data):
    # if we’re not on the viz tab, clear both graph and any previous error
    if tab != "tab-viz" or store_data is None:
        return [], ""

    try:
        # build your nodes
        nodes = [{"data": {"id": b, "label": b}} for b in store_data.keys()]

        # build your edges
        strength = "High" if np.random.rand() < 0.6 else "Average"
        raw_edges = extract_relations(conversation_history)
        edges = [
            {"data": {"source": src, "target": tgt, "relation": rel, "strength": strength}}
            for src, tgt, rel in raw_edges
            if src in store_data and tgt in store_data
        ]

        # no error to report
        return nodes + edges, ""

    except Exception:
        import traceback
        tb = traceback.format_exc()
        # return an empty graph plus the full traceback in the UI
        return [], tb


@app.callback(
    Output("edge-info","children"),
    Input("bot-graph","tapEdgeData")
)
def display_edge(edge):
    if not edge:
        return "Click an edge to see details."
    return (
        f"Relation: {edge['relation']}\n"
        f"Strength: {edge.get('strength','?')}"
    )
    
# 3) Callback to show chat‐history when you click a node
@app.callback(
    Output("node-info", "children"),
    Input("bot-graph", "tapNodeData"),
    State("chat-histories", "data")
)
def display_node_details(node_data, store_data):
    if not node_data:
        return "Click a bot to see its recent messages."
    bot = node_data["id"]
    history = store_data.get(bot, [])
    lines = [f"{m['sender'].upper()}: {m['text']}" for m in history[-10:]]  # last 10 msgs
    return "\n".join(lines)
    
# --- Polling callback: refill the store whenever conversation_history changes ---
@app.callback(
    Output("chat-histories", "data"),
    Input("refresh-interval", "n_intervals"),
    State("chat-histories", "data")
)
def refresh_histories(n, current_store):
    new_store = map_conversation_history_to_dash(conversation_history)
    if new_store != current_store:
        return new_store
    raise PreventUpdate

# --- Show/hide the Bot List tabs on viz vs chat ---
@app.callback(
    Output("bot-selector", "style"),
    Input("main-tabs", "value")
)
def toggle_bot_list(tab):
    return {"display":"none"} if tab=="tab-viz" else {"display":"block"}

# --- Populate the Bot List from chat-histories store ---
@app.callback(
    Output("bot-selector", "children"),
    Input("chat-histories", "data")
)
def update_bot_selector(histories):
    if histories is None:
        return []
    return [
        dcc.Tab(
            label=bot, value=bot,
            style={
                "backgroundColor": "#131313",
                "color": "#888",
                "padding": "8px 16px",
                "display": "block"    # ensure each one is its own line
            },
            selected_style={
                "backgroundColor": "#1e1e1e",
                "color": "#fff",
                "fontWeight": "bold"
            }
        )
        for bot in histories.keys()
    ]

# --- Swap content based on main‑tabs ---
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value")
)
def render_tab_content(tab):
    return make_chat_ui() if tab=="tab-chat" else viz_layout

# --- Chat send callback: update conversation_history and clear input ---
@app.callback(
    Output("user-input",      "value"),
    Input("send-button",      "n_clicks"),
    State("user-input",       "value"),
    State("bot-selector",     "value"),
    prevent_initial_call=True
)
def on_send(n_clicks, message, current_bot):
    if not message or not message.strip():
        raise PreventUpdate

    # Append to global conversation_history
    conv = conversation_history.get(current_bot, "")
    conv += f"<|im_start|>user\n{message}<|im_end|>"
    reply = chat(message, current_bot)
    conv += f"<|im_start|>{current_bot}\n{reply}<|im_end|>"
    conversation_history[current_bot] = conv

    # Clear the input box; the Interval callback will refresh the store
    return ""

# --- Update chat window when bot or history changes ---
@app.callback(
    Output("chat-window", "children"),
    Input("bot-selector",   "value"),
    Input("chat-histories", "data")
)
def update_chat_window(current_bot, histories):
    bot_hist = histories.get(current_bot, []) if histories else []
    if not bot_hist:
        return html.Div("No messages yet…", style={"color":"#888"})
    bubbles = []
    for m in bot_hist:
        is_user = (m["sender"]=="user")
        bubbles.append(html.Div(
            m["text"],
            style={
                "alignSelf":  "flex-end" if is_user else "flex-start",
                "background": "#142cb5" if is_user else "#4c4c4c",
                "color":      "#fff",
                "padding":    "8px 12px",
                "borderRadius":"12px",
                "margin":     "4px",
                "maxWidth":   "75%",
                "boxSizing":  "border-box"
            }
        ))
    return bubbles

def map_conversation_history_to_dash(
    conversation_history: Dict[str, str]
) -> Dict[str, List[Dict[str, str]]]:
    """
    Strips the first system block, then parses all user/assistant blocks—
    including those missing an explicit label (treated as assistant)—
    into a list of {"sender": "...", "text": "..."} dicts.
    """
    dash_histories: Dict[str, List[Dict[str, str]]] = {}

    # 1) remove only the very first system block
    sys_re = re.compile(
        r"<\|im_start\|>system\n.*?<\|im_end\|>", 
        re.DOTALL
    )

    # 2) capture user/assistant or unlabeled blocks
    msg_re = re.compile(
        r"<\|im_start\|>"
        r"(?:(user|assistant)\n)?"  # optional role
        r"(.*?)"                    # message body
        r"<\|im_end\|>", 
        re.DOTALL
    )

    for bot, conv in conversation_history.items():
        # strip system prompt
        body = sys_re.sub("", conv, count=1).strip()

        entries: List[Dict[str,str]] = []
        for role, text in msg_re.findall(body):
            sender = role if role in ("user","assistant") else bot
            txt = text.strip()
            if not txt:
                continue
            entries.append({"sender": sender, "text": txt})

        dash_histories[bot] = entries

    return dash_histories


def extract_relations(conversation_history):
    edges = []
    for bot, full_text in conversation_history.items():
        # find all your X is Y occurrences
        for rel, other in rel_re.findall(full_text):
            edges.append((bot, other, rel.lower()))
    return edges

url = "http://127.0.0.1:8050"
threading.Timer(1, lambda: webbrowser.open_new_tab(url)).start()
app.run_server(mode='external',debug=False)
