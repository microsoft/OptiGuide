from re import S
import streamlit as st
import asyncio
from openai import OpenAI
from optiguide import OptiGuideAgent
import autogen
from autogen.agentchat import Agent, UserProxyAgent
import streamlit.components.v1 as components

config_list = [
    # {
    #     "model": "gpt-4o",
    #     "api_key": os.getenv("OPENAI_API_KEY"),
    # },
    {
        "model": "phi-4",
        "api_key": "ollama",
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
        m.addConstr(x[s,'roastery2'] <= capacity_in_supplier[s] * z[s], "_")
    ```

    ----------
    Question: What if there's a 13% jump in the demand for light coffee at cafe1?
    Answer Code:
    ```python
    light_coffee_needed_for_cafe["cafe1"] = light_coffee_needed_for_cafe["cafe1"] * (1 + 13/100)
    ```

    ----------
    Question: Remove cafe1
    Answer Code:
    ```python
    shipping_cost_from_roastery_to_cafe = {k:v for k, v in shipping_cost_from_roastery_to_cafe.items() if k[1] != 'cafe1'}
    light_coffee_needed_for_cafe.pop('cafe1')
    dark_coffee_needed_for_cafe.pop('cafe1')
    cafes = [c for c in cafes if c != 'cafe1']
    ```
    """

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
<div id="slider"></div>
<div id="graph"></div>
<script src="https://d3js.org/d3.v7.js"></script>
<script src="https://unpkg.com/@hpcc-js/wasm@2/dist/graphviz.umd.js" type="javascript/worker"></script>
<script src="https://unpkg.com/d3-graphviz@5/build/d3-graphviz.js"></script>
<script src="https://unpkg.com/d3-simple-slider"></script>
    <style>
        .node:hover {{
            filter: drop-shadow( 3px 3px 2px rgba(0, 0, 0, .2));
        }}

        .node text {{
            font: 30px sans-serif;
            font-weight: bold;
            stroke: black;
            stroke-width: 1px;
            fill: white;
        }}

        #myslider {{ width: 100%; }}
    </style>
<script>
    const scale = 0.8;
    width = window.frameElement.clientWidth;
    height = window.frameElement.clientHeight;

    function attributer(datum, index, nodes) {{
        var selection = d3.select(this);
        if (datum.tag == "svg") {{
            datum.attributes = {{
                ...datum.attributes,
                width: width,
                height: height-25,
            }};
        }}
    }}

    const iframe = window.frameElement;
    dotString = [{",".join(['`' + g + '`' for g in st.session_state['visual_history']])}];

    G = d3.select("#graph").graphviz()
          .attributer(attributer)
          .addImage("app/static/supplier.png", "40px", "55px")
          .addImage("app/static/roaster.png", "60px", "50px")
          .addImage("app/static/cafe.png", "45px", "80px")
          .renderDot(dotString[dotString.length-1]);

    function showVal(val) {{
        G.transition(function () {{
            return d3.transition()
                .duration(500);
        }}).renderDot(dotString[val]);
    }}

    d3.select("#myslider")
      .attr("max", dotString.length - 1)
      .attr("value", dotString.length - 1)
      .attr('width', width-100)
      .on("input", function() {{ showVal(d3.select(this).property("value")); }});

    function handleResize() {{
        width = iframe.clientWidth;
        height = iframe.clientHeight;

        d3.select("#myslider").attr('width', width-100);
        d3.select('#graph svg').attr('width', width).attr('height', height-25);
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
    chat_box = st.container(border=True, height=800)
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
