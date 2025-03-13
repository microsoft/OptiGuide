from re import S
import streamlit as st
import asyncio
from openai import OpenAI
from optiguide import OptiGuideAgent
import autogen
from autogen.agentchat import Agent, UserProxyAgent
import streamlit.components.v1 as components

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import json
import os

config_list = [
    # {
    #     "model": "gpt-4o",
    #     "api_key": os.getenv("OPENAI_API_KEY"),
    # },
    {
        "model": "phi-4",
        "api_key": "empty",
        "base_url": "http://localhost:8080/v1",
        "max_retries": 10,
        "price" : [0, 0]
    }
]

st.set_page_config( # this line should be above all other ST codes.
    page_title="OptiGuide Chat",
    page_icon="img/optiguide_logo_s.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Coffee Supply Chain Optimization")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(base_url="http://localhost:8080/v1", api_key="empty")

if "loop" not in st.session_state:
    st.session_state['loop'] = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state['loop'])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "phi-4-gguf"

if "visual_history" not in st.session_state:
    st.session_state["visual_history"] = []

if "visual_current" not in st.session_state:
    st.session_state["visual_current"] = ""

# load the source code from coffee.py
if "src_code" not in st.session_state:
    with open(r"coffee.py", "r") as f:
        st.session_state["src_code"] = f.read()

if "api_code" not in st.session_state:
    with open(r"coffee_instance.pyi", "r") as f:
        st.session_state["api_code"] = f.read()

if "tmp_src_code" not in st.session_state:
    st.session_state['tmp_src_code'] = st.session_state['src_code']

if 'prompts' not in st.session_state:
    st.session_state['prompts'] = """
    ----------
    Question: Why is it not recommended to use just one supplier for roastery 2?
    Answer Code:
    ```python
    z = m.addVars(suppliers, vtype=GRB.BINARY, name="z")
    m.addConstr(sum(z[s] for s in suppliers) <= 1, "_")
    for s in suppliers:
        m.addConstr(x[s,'roastery2'] <= instance.capacity_in_supplier[s] * z[s], "_")
    ```

    ----------
    Question: What if there's a 13% jump in the demand for light coffee at cafe1?
    Answer Code:
    ```python
    instance.light_coffee_needed_for_cafe["cafe1"] *= (1 + 13/100)
    ```

    ----------
    Question: Remove roastery2
    Answer Data:
    ```python
    instance.delete_roastery('roastery2')
    ```

    ----------
    Question: Add a cafe called javabrew with light coffee needed of 20 and dark coffee needed of 30, and connect all roasteries to it with a shipping cost of 5.
    Answer Data:
    ```python
    instance.add_cafe('javabrew', 20, 30)

    for r in instance.roasteries:
        instance.connect_roastery_to_cafe(r, 'javabrew', 5)
    ```

    ----------
    Question: Update capacity of supplier1 to 100
    Answer Data:
    ```python
    instance.update_supplier('supplier1', 100)

    for r in instance.roasteries:
        instance.connect_roastery_to_cafe(r, 'javabrew', 5)
    ```

    ----------
    Question: Change cost of burner1 to 3.5 for dark coffee.
    Answer Data:
    ```python
    instance.update_roastery('burner1', dark_cost=3.5)
    ```

    No other code needed.
    """

path = os.path.dirname(os.path.realpath(__file__))

if "rag" not in st.session_state:
    rag = QdrantClient(":memory:") # Create in-memory Qdrant instance
    encoder = SentenceTransformer("thenlper/gte-small")

    st.session_state.rag = rag
    st.session_state.encoder = encoder

    # load the datastore
    rag.create_collection(
        collection_name="example_qa",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
            distance=models.Distance.COSINE
        )
    )

    with open(path + '/../QAs/coffee.benchmark.json') as f:
        d = json.load(f)
    
    data = [{'q': x['QUESTION'], 'a': x['DATA CODE'] if 'DATA CODE' in x else x['CONSTRAINT CODE']} for x in d['questions']]

    rag.upload_points(
        collection_name="example_qa",
        points=[
            models.PointStruct(
                id=idx,
                vector=encoder.encode(doc["q"]).tolist(),
                payload=doc
            ) for idx, doc in enumerate(data)
        ]
    )

    # # quick test
    # hits = rag.query_points(
    #     collection_name="example_qa",
    #     query=encoder.encode("What are the possible consequences if cafe cafe2's demand grows by 29%?").tolist(),
    #     limit=3
    # )
    # for hit in hits.points:
    #   print(hit.payload, "score:", hit.score)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "debug_messages" not in st.session_state:
    st.session_state['debug_messages'] = []

visualization = st.container(border=False)

chat_tab, template_tab, prompt_tab, code_tab, debug_tab= st.tabs([
    "Chat",
    "Template",
    "Prompt",
    "Code",
    "Debug",
])

class TrackableAssistantAgent(OptiGuideAgent):
    def _process_received_message(self, message, sender, silent):
        if sender.name != "user":
            st.session_state['debug_messages'].append({"role": sender.name, "content": message})
        # with debug_tab:
        #     with st.chat_message(sender.name):
        #         st.markdown(message)
        #     #     #st.session_state.messages.append({"role": "assistant", "content": message})
        return super()._process_received_message(message, sender, silent)

class TrackableUserProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        # st.session_state['debug_messages'].append({"role": "assistant", "content": message})

        # with debug_tab:
        #     with st.chat_message(sender.name):
        #         st.markdown(message)
        #     #st.session_state.messages.append({"role": "user", "content": message})

        return super()._process_received_message(message, sender, silent)

def code_callback(message):
    st.session_state['tmp_src_code'] = message

    with code_tab:
        st.code(message)

def visual_callback(message):
    st.session_state['visual_history'].append(message)
    
    d3_code = f'''
<input id="myslider" type="range" min="0" max="0" step="1" />
<div id="graph"></div>
<script src="https://d3js.org/d3.v7.js"></script>
<script src="https://unpkg.com/@hpcc-js/wasm@2/dist/graphviz.umd.js" type="javascript/worker"></script>
<script src="https://unpkg.com/d3-graphviz@5/build/d3-graphviz.js"></script>
<script src="https://unpkg.com/d3-simple-slider"></script>
    <style>
        .node:hover {{
            filter: drop-shadow( 3px 3px 2px rgba(0, 0, 0, .2));
        }}

        .node {{
            cursor: default;
        }}

        #myslider {{ width: 100%; opacity: 0.2; }}
        #myslider:hover {{ opacity: 1; }}

        input[type=range] {{-webkit-appearance: none;margin: 18px 0;width: 100%;}}
        input[type=range]:focus {{outline: none;}}
        input[type=range]::-webkit-slider-runnable-track {{width: 100%;height: 16px;cursor: pointer;box-shadow: 1px 1px 1px #000000, 0px 0px 1px #0d0d0d;background: #dbdbdb;border-radius: 1.3px;border: 0.2px solid #010101;}}
        input[type=range]::-webkit-slider-thumb {{box-shadow: 1px 1px 1px #000000, 0px 0px 1px #0d0d0d;border: 1px solid #000000;height: 36px;width: 16px;border-radius: 3px;background: #ffffff;cursor: pointer;-webkit-appearance: none;margin-top: -11px;}}
        input[type=range]:focus::-webkit-slider-runnable-track {{background: #367ebd;}}
        input[type=range]::-moz-range-track {{width: 100%;height: 14px;cursor: pointer;box-shadow: 1px 1px 1px #000000, 0px 0px 1px #0d0d0d;background: #dbdbdb;border-radius: 1.3px;border: 0.2px solid #010101;}}
        input[type=range]::-moz-range-thumb {{box-shadow: 1px 1px 1px #000000, 0px 0px 1px #0d0d0d;border: 1px solid #000000;height: 36px;width: 16px;border-radius: 3px;background: #ffffff;cursor: pointer;}}
        input[type=range]::-ms-track {{width: 100%;height: 12px;cursor: pointer;background: transparent;border-color: transparent;border-width: 16px 0;color: transparent;}}
        input[type=range]::-ms-fill-lower {{background: #2a6495;border: 0.2px solid #010101;border-radius: 2.6px;box-shadow: 1px 1px 1px #000000, 0px 0px 1px #0d0d0d;}}
        input[type=range]::-ms-fill-upper {{background: #dbdbdb;border: 0.2px solid #010101;border-radius: 2.6px;box-shadow: 1px 1px 1px #000000, 0px 0px 1px #0d0d0d;}}
        input[type=range]::-ms-thumb {{box-shadow: 1px 1px 1px #000000, 0px 0px 1px #0d0d0d;border: 1px solid #000000;height: 36px;width: 16px;border-radius: 3px;background: #ffffff;cursor: pointer;}}
        input[type=range]:focus::-ms-fill-lower {{background: #dbdbdb;}}
        input[type=range]:focus::-ms-fill-upper {{background: #367ebd;}}
    </style>
<script>
    width = window.frameElement.clientWidth;
    height = window.frameElement.clientHeight;

    function attributer(datum, index, nodes) {{
        if (datum.tag == "svg") {{
            datum.attributes = {{
                ...datum.attributes,
                width: width,
                height: height-50,
            }};
        }}
        else if (datum.tag == "image") {{
            datum.attributes = {{
                ...datum.attributes,
                preserveAspectRatio: "xMidYMid"
            }};
        }}
    }}

    const iframe = window.frameElement;
    dotString = [{",".join(['`' + g + '`' for g in st.session_state['visual_history']])}];

    function transitionFactory() {{
        return d3.transition("main")
            .ease(d3.easeCubic)
            .duration(300);
    }}

    G = d3.select("#graph").graphviz()
          .attributer(attributer)
          .addImage("app/static/supplier.png", "60px", "60px")
          .addImage("app/static/roaster.png", "60px", "60px")
          .addImage("app/static/cafe.png", "60px", "60px")
          .tweenShapes(false)
          .fade(true)          
          .renderDot(dotString[dotString.length-1]);

    function showVal(val) {{
        G.transition(transitionFactory).renderDot(dotString[val]);
    }}

    d3.select("#myslider")
      .attr("max", dotString.length - 1)
      .attr("value", dotString.length - 1)
      .on("input", function() {{ showVal(d3.select(this).property("value")); }});

    function handleResize() {{
        width = iframe.clientWidth;
        height = iframe.clientHeight;

        d3.select('#graph svg').attr('width', width).attr('height', height-50);
    }}

    handleResize();
    d3.select(window).on('resize', handleResize);
</script>
'''
    st.session_state["visual_current"] = d3_code
    
    #with visualization:
        #st.graphviz_chart(message)
        #components.html("d3_code", height=400)

if "agent" not in st.session_state:
    st.session_state['agent'] = TrackableAssistantAgent(
        name="assistant",
        source_code=st.session_state['src_code'],
        doc_str=st.session_state['api_code'],
        debug_times=1,
        example_qa=st.session_state['prompts'],
        llm_config={
            "seed": 42,
            "config_list": config_list,
        },
        visual_cb=visual_callback,
        code_cb=code_callback
    )

if "user" not in st.session_state:
    st.session_state['user'] = TrackableUserProxyAgent(
        "user", max_consecutive_auto_reply=0,
        human_input_mode="NEVER", code_execution_config=False
    )


with template_tab:
    edit_code_button = st.popover("Edit Code", use_container_width=True)

    with edit_code_button:
        edited_code = st.text_area("Edit your code below:",
                                   value=st.session_state['src_code'],
                                   height=300)
        if st.button("Save Code"):
            st.session_state['src_code'] = edited_code
            st.success("Code updated successfully!")

            st.session_state['agent'] = TrackableAssistantAgent(
                name="assistant",
                source_code=st.session_state['src_code'],
                doc_str=st.session_state['api_code'],
                debug_times=1,
                example_qa=st.session_state['prompts'],
                llm_config={
                    "seed": 42,
                    "config_list": config_list,
                },
                visual_cb=visual_callback,
                code_cb=code_callback
            )

    st.code(st.session_state['src_code'])


with prompt_tab:
    edit_prompt_button = st.popover("Edit Prompts", use_container_width=True)

    with edit_prompt_button:
        edited_prompt = st.text_area("Edit your prompt below:",
                                value=st.session_state['prompts'],
                                height=300)
        if st.button("Save Prompt"):
            st.session_state['prompts'] = edited_prompt
            st.success("Prompts updated successfully!")

            st.session_state['agent'] = TrackableAssistantAgent(
                name="assistant",
                source_code=st.session_state['src_code'],
                doc_str=st.session_state['api_code'],
                debug_times=1,
                example_qa=st.session_state['prompts'],
                llm_config={
                    "seed": 42,
                    "config_list": config_list,
                },
                visual_cb=visual_callback,
                code_cb=code_callback
            )

    st.code(st.session_state['prompts'])

with chat_tab:
    chat_box = st.container(border=True, height=300)
    text_box = st.container(border=False)

    # Accept user input
    with text_box:
        if prompt := st.chat_input("What would you like to check?"):
            async def initiate_chat():
                await st.session_state['user'].a_initiate_chat(st.session_state['agent'], message=prompt)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state['debug_messages'].append({"role": "user", "content": prompt})

            st.session_state['agent'].set_sourcecode(st.session_state['tmp_src_code'])  # use the updated source code

            print(prompt)
            hits = st.session_state.rag.query_points(
                collection_name="example_qa",
                query=st.session_state.encoder.encode(prompt).tolist(),  # encode last message only
                limit=3
            )

            print("SUCCESSFUL LOOKUP!")
            st.session_state['agent']._example_qa = "-------------------------------\n".join(
                [f"Question: {x.payload['q']}\nAnswer Data: ```python\n{x.payload['a']}\n```\n" for x in hits.points]) + "\n\nNo other code needed.\n"
            print(st.session_state['agent']._example_qa)

            response = st.session_state['user'].initiate_chat(st.session_state['agent'], message=prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.summary})
            st.session_state['debug_messages'].append({"role": "assistant", "content": response.summary})

            # # Display user message in chat message container
            # with chat_box:
            #     with st.chat_message("user"):
            #         st.markdown(prompt)
            #st.session_state['loop'].run_until_complete(initiate_chat())

    with chat_box:
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

            # with chat_box:
            #     with st.chat_message("assistant"):
            #         st.markdown(response.summary)
            #         st.session_state.messages.append({"role": "assistant", "content": response.summary})

            #         # with st.chat_message("assistant"):
            #         #     #loop.run_until_complete(initiate_chat())
            #         #     response = user.initiate_chat(agent, message=prompt)

            #             # stream = client.chat.completions.create(
            #             #     model=st.session_state["openai_model"],
            #             #     messages=[
            #             #         {"role": m["role"], "content": m["content"]}
            #             #         for m in st.session_state.messages
            #             #     ],
            #             #     stream=True,
            #             # )
            #             # response = st.write_stream(stream)
            #         #st.session_state.messages.append({"role": "assistant", "content": response.summary})

# with code_tab:
#     st.code(st.session_state['tmp_src_code'])

with visualization:
    components.html(st.session_state['visual_current'], height=400)

with debug_tab:
    debug_chat_box = st.container(border=True, height=800)

    with debug_chat_box:
        for message in st.session_state['debug_messages']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
