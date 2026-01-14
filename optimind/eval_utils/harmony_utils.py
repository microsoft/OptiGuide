# A majority of this file is adapted from the vLLM project, which is licensed under the Apache 2.0 license.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 
import datetime
import re
import os
import json
from collections.abc import Iterable, Sequence
from typing import Literal, Optional, Union

from openai.types.responses import (ResponseFunctionToolCall,
                                    ResponseOutputItem, ResponseOutputMessage,
                                    ResponseOutputText, ResponseReasoningItem)
from openai.types.responses.response_function_web_search import (
    ActionFind, ActionOpenPage, ActionSearch, ResponseFunctionWebSearch)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent)
from openai.types.responses.tool import Tool
from openai_harmony import (Author, Conversation, DeveloperContent,
                            HarmonyEncodingName, Message, ReasoningEffort,
                            Role, StreamableParser, SystemContent, TextContent,
                            ToolDescription, load_harmony_encoding,
                            RenderConversationConfig)
from json_repair import repair_json
from uuid import uuid4 as random_uuid

REASONING_EFFORT = {
    "high": ReasoningEffort.HIGH,
    "medium": ReasoningEffort.MEDIUM,
    "low": ReasoningEffort.LOW,
}

_harmony_encoding = None


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def get_system_message(
    model_identity: Optional[str] = None,
    reasoning_effort: Optional[Literal["high", "medium", "low"]] = None,
    start_date: Optional[str] = None,
    browser_description: Optional[str] = None,
    python_description: Optional[str] = None,
) -> Message:
    sys_msg_content = SystemContent.new()
    if model_identity is not None:
        sys_msg_content = sys_msg_content.with_model_identity(model_identity)
    if reasoning_effort is not None:
        sys_msg_content = sys_msg_content.with_reasoning_effort(
            REASONING_EFFORT[reasoning_effort])
    if start_date is None:
        # NOTE(woosuk): This brings non-determinism in vLLM. Be careful.
        start_date = datetime.datetime.now().strftime("%Y-%m-%d")
    sys_msg_content = sys_msg_content.with_conversation_start_date(start_date)
    if browser_description is not None:
        sys_msg_content = sys_msg_content.with_tools(browser_description)
    if python_description is not None:
        sys_msg_content = sys_msg_content.with_tools(python_description)
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sys_msg_content)
    return sys_msg


def get_developer_message(instructions: Optional[str] = None,
                          tools: Optional[list[Tool]] = None) -> Message:
    dev_msg_content = DeveloperContent.new()
    if instructions is not None:
        dev_msg_content = dev_msg_content.with_instructions(instructions)
    if tools is not None:
        function_tools = []
        for tool in tools:
            if tool.type in ("web_search_preview", "code_interpreter"):
                # These are built-in tools that are added to the system message.
                pass
            elif tool.type == "function":
                function_tools.append(tool)
            else:
                raise ValueError(f"tool type {tool.type} not supported")
        if function_tools:
            function_tool_descriptions = [
                ToolDescription.new(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                ) for tool in function_tools
            ]
            dev_msg_content = dev_msg_content.with_function_tools(
                function_tool_descriptions)
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_msg_content)
    return dev_msg


def get_user_message(content: str) -> Message:
    return Message.from_role_and_content(Role.USER, content)


def parse_chat_input(chat_msg) -> Message:
    role = chat_msg["role"]
    content = chat_msg["content"]
    if isinstance(content, str):
        contents = [TextContent(text=content)]
    else:
        # TODO: Support refusal.
        contents = [TextContent(text=c["text"]) for c in content]
    msg = Message.from_role_and_contents(role, contents)
    return msg


def render_for_completion(messages: list[Message]) -> list[int]:
    conversation = Conversation.from_messages(messages)
    token_ids = get_encoding().render_conversation_for_completion(
        conversation, Role.ASSISTANT)
    return token_ids


def parse_output_message(message: Message) -> list[ResponseOutputItem]:
    """
    Parse a Harmony message into a list of output response items.
    """
    if message.author.role != "assistant":
        # This is a message from a tool to the assistant (e.g., search result).
        # Don't include it in the final output for now. This aligns with
        # OpenAI's behavior on models like o4-mini.
        return []

    output_items: list[ResponseOutputItem] = []
    recipient = message.recipient
    content_type = message.content_type

    if recipient is not None and recipient.startswith("browser."):
        if len(message.content) != 1:
            raise ValueError("Invalid number of contents in browser message")
        content = message.content[0]
        try:
            browser_call = json.loads(content.text)
        except:
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(
                            text=content.text, type="reasoning_text"
                        )
                    ],
                    status=None,
                )
                output_items.append(reasoning_item)
            return output_items
        
        if recipient == "browser.search":
            action = ActionSearch(
                query=f"cursor:{browser_call.get('query', '')}", type="search")
        elif recipient == "browser.open":
            action = ActionOpenPage(
                url=f"cursor:{browser_call.get('url', '')}", type="open_page")
        elif recipient == "browser.find":
            action = ActionFind(pattern=browser_call["pattern"],
                                url=f"cursor:{browser_call.get('url', '')}",
                                type="find")
        else:
            raise ValueError(f"Unknown browser action: {recipient}")
        web_search_item = ResponseFunctionWebSearch(
            id=f"ws_{random_uuid()}",
            action=action,
            status="completed",
            type="web_search_call",
        )
        output_items.append(web_search_item)            
    elif message.channel == "analysis":
        for content in message.content:
            reasoning_item = ResponseReasoningItem(
                id=f"rs_{random_uuid()}",
                summary=[],
                type="reasoning",
                content=[
                    ResponseReasoningTextContent(text=content.text,
                                                 type="reasoning_text")
                ],
                status=None,
            )
            output_items.append(reasoning_item)
    elif message.channel == "commentary":
        if recipient is None:
            # add to reasoning
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(text=content.text,
                                                     type="reasoning_text")
                    ],
                    status=None,
                )
                output_items.append(reasoning_item)
        elif recipient.startswith("functions"):  # functions.
            function_name = recipient.split(".")[-1] if "." in recipient else recipient.replace("functions", "").strip()
            if function_name == "" and ("code" in content_type or "python" in content_type):
                function_name = "code"
                
            for content in message.content:
                arguments = content.text
                local_function_name = function_name
                if not function_name:
                    try:
                        parsed_args = json.loads(arguments)
                    except:
                        try:
                            parsed_args = json.loads(repair_json(arguments))
                        except:
                            parsed_args = {}
                    
                    if isinstance(parsed_args, dict) and "name" in parsed_args and "arguments" in parsed_args:
                        local_function_name = parsed_args["name"]
                        arguments = parsed_args["arguments"]
                        if not isinstance(arguments, str):
                            arguments = json.dumps(arguments)
                        
                random_id = random_uuid()
                response_item = ResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=f"call_{random_id}",
                    type="function_call",
                    name=local_function_name,
                    id=f"ft_{random_id}",
                )
                output_items.append(response_item)
        elif recipient.startswith("python") or message.recipient.startswith("browser") \
            or recipient.startswith("container"):
            print("Recipient is", recipient, "!!!")
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(text=content.text,
                                                     type="reasoning_text")
                    ],
                    status=None,
                )
                output_items.append(reasoning_item)
        else:
            # Unknown / missing recipient: don't crash, just treat as reasoning
            for content in message.content:
                reasoning_item = ResponseReasoningItem(
                    id=f"rs_{random_uuid()}",
                    summary=[],
                    type="reasoning",
                    content=[
                        ResponseReasoningTextContent(
                            text=content.text, type="reasoning_text"
                        )
                    ],
                    status=None,
                )
                output_items.append(reasoning_item)
            # raise ValueError(f"Unknown recipient: {recipient}")
    elif message.channel == "final":
        contents = []
        for content in message.content:
            output_text = ResponseOutputText(
                text=content.text,
                annotations=[],  # TODO
                type="output_text",
                logprobs=None,  # TODO
            )
            contents.append(output_text)
        text_item = ResponseOutputMessage(
            id=f"msg_{random_uuid()}",
            content=contents,
            role=message.author.role,
            # if the parser still has messages (ie if the generator got cut
            # abruptly), this should be incomplete
            status="incomplete",
            type="message",
        )
        output_items.append(text_item)
    else:
        raise ValueError(f"Unknown channel: {message.channel}")
    return output_items


def parse_remaining_state(
        parser: StreamableParser) -> list[ResponseOutputItem]:
    if not parser.current_content:
        return []
    if parser.current_role != Role.ASSISTANT:
        return []
    current_recipient = parser.current_recipient
    if (current_recipient is not None
            and current_recipient.startswith("browser.")):
        return []

    if parser.current_channel == "analysis":
        reasoning_item = ResponseReasoningItem(
            id=f"rs_{random_uuid()}",
            summary=[],
            type="reasoning",
            content=[
                ResponseReasoningTextContent(text=parser.current_content,
                                             type="reasoning_text")
            ],
            status=None,
        )
        return [reasoning_item]
    elif parser.current_channel == "final":
        output_text = ResponseOutputText(
            text=parser.current_content,
            annotations=[],  # TODO
            type="output_text",
            logprobs=None,  # TODO
        )
        text_item = ResponseOutputMessage(
            id=f"msg_{random_uuid()}",
            content=[output_text],
            role="assistant",
            status="completed",
            type="message",
        )
        return [text_item]
    return []


def get_stop_tokens_for_assistant_actions() -> list[int]:
    return get_encoding().stop_tokens_for_assistant_actions()


def get_streamable_parser_for_assistant() -> StreamableParser:
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)


def parse_output_into_messages(token_ids: Iterable[int]) -> StreamableParser:
    parser = get_streamable_parser_for_assistant()
    for token_id in token_ids:
        parser.process(token_id)
    return parser


def parse_chat_output(token_ids: Sequence[int]) -> tuple[str | None, str | None, bool]:
    parser = parse_output_into_messages(token_ids)
    output_msgs = parser.messages
    is_tool_call = False  # TODO: update this when tool call is supported

    if len(output_msgs) == 0:
        # The generation has stopped during reasoning.
        reasoning = parser.current_content
        final_content = None
    elif len(output_msgs) == 1:
        # The generation has stopped during final message.
        reasoning = output_msgs[0].content[0].text
        final_content = parser.current_content
    else:
        reasoning_msg = output_msgs[:-1]
        final_msg = output_msgs[-1]
        reasoning = "\n".join([msg.content[0].text for msg in reasoning_msg])
        final_content = final_msg.content[0].text
    return reasoning, final_content, is_tool_call



def sanitize_harmony_headers_keep_tools(token_txt: str) -> str:
    """
    Normalize legacy / non-Harmony headers into something the Harmony parser
    understands, while PRESERVING tool calls.
    """
    # 0) Normalize all <|return|> (end of inference) to <|end|> (end of assistant turn)
    token_txt = token_txt.replace("<|return|>", "<|end|>")
    # Strip legacy / unsupported control tokens <|call|><|end|> -> <|end|> (strip empty calls)
    token_txt = re.sub(r"<\|call\|>\s*<\|end\|>", "<|end|>", token_txt)

    # 1) Legacy "final" header -> assistant final
    token_txt = token_txt.replace(
        "<|start|>final<|message|>",
        "<|start|>assistant<|channel|>final<|message|>",
    )
    
    # <|start|>assistant<|channel|>analysis to=functions. -> <|start|>assistant<|channel|>commentary to=functions.
    token_txt = re.sub(
        r"<\|start\|>assistant<\|channel\|>analysis\s+to=(?P<recipient>functions[^\s<]+)",
        r"<|start|>assistant<|channel|>commentary to=\g<recipient>",
        token_txt,
    )
    
    # <|start|>assistant<|channel|>commentary to=functions. -> <|start|>assistant<|channel|>commentary to=functions.
    token_txt = re.sub(
        r"<\|start\|>assistant\s*<\|channel\|>commentary\s*to=functions\s+(\w+)(\s*<\|message\|>)",
        r"<|start|>assistant<|channel|>commentary to=functions.\1\2",
        token_txt,
    )

    # 3) Assistant headers that completely lack a channel:
    token_txt = re.sub(
        r"<\|start\|>assistant\s*<\|message\|>",
        "<|start|>assistant<|channel|>analysis<|message|>",
        token_txt,
    )
    
    # 3b) Ensure <|channel|>analysis / final are followed by <|message|>
    token_txt = re.sub(
        r"(<\|channel\|>\s*(analysis|final))(?!\s*<\|message\|>)",
        r"\1<|message|>",
        token_txt,
    )
        
    # 3c) Commentary: insert <|message|> only if NOT followed by <|message|>, 'to=' or <|constrain|>
    # This avoids breaking legacy "commentary to=..." or "commentary <|constrain|>..."
    token_txt = re.sub(
        r"(<\|channel\|>\s*commentary)(?!\s*(<\|message\||to=|<\|constrain\|>))", r"\1<|message|>",
        token_txt,
    )
    
    # 3d) Commentary: allow optional "to=..." and "<|constrain|>...", then ensure there's a <|message|>
    def _insert_message_if_missing(m: re.Match) -> str:
        header = m.group(1)
        end = m.end(1)
        # Look at what actually follows in the original string
        if token_txt[end:].lstrip().startswith("<|message|>"):
            return header  # already correct
        return header + "<|message|>"
    
    token_txt = re.sub(
        r'(<\|channel\|>\s*commentary(?:\s+to=[^\s<]+)?(?:\s*<\|constrain\|>[^\s<]+)?)',
        _insert_message_if_missing,
        token_txt,
        flags=re.DOTALL,
    )
    
    # 4) normalize all <|return|> to <|end|>, and remove everything after the last <|end|>
    last_end_idx = token_txt.rfind("<|end|>")
    if last_end_idx != -1:
        token_txt = token_txt[: last_end_idx + len("<|end|>")]
    else:
        # If there's no <|end|>, ensure each assistant message is properly closed
        token_txt = token_txt + "<|end|>"

    return token_txt


def sanitize_start_end_message(prefix_txt: str) -> str:
    msg = prefix_txt.lstrip()
    msg = msg.replace("<|return|>", "<|end|>")
    if not msg.endswith("<|end|>"):
        msg += "<|end|>"
        
    # Heuristic: looks like a message header fragment
    if msg.startswith("<|channel|>"):
        msg = "<|start|>assistant" + msg
        return msg
    # Optionally: support other incomplete headers here
    return msg


def extract_tool_args(raw: str) -> str:
    if raw is None:
        return ""

    s = raw.strip()

    # 1) Split after first <|message|> if it exists
    marker = "<|message|>"
    idx = s.find(marker)
    if idx != -1:
        end_idx = idx + len(marker)
        s = s[end_idx:].strip()
    else:
        # 2) Otherwise, start from first '{' if present
        brace_idx = s.find("{")
        if brace_idx != -1:
            s = s[brace_idx:].strip()

    # 3) Trim after the last closing brace if there is one
    last_brace_idx = s.rfind("}")
    if last_brace_idx != -1:
        s = s[: last_brace_idx + 1].strip()
    
    # 4) Balance braces: if we have more '{' than '}', append missing '}'
    open_braces = s.count("{")
    close_braces = s.count("}")
    if open_braces > close_braces:
        s = s + ("}" * (open_braces - close_braces))
        
    return s, open_braces

def save_arguments(arguments, prefix="tmp_args"):
    subdir = os.path.join("tmp_harmony", prefix)
    os.makedirs(subdir, exist_ok=True)
    save_idx = 0
    while os.path.exists(os.path.join(subdir, f"{save_idx}.json")):
        save_idx += 1
    with open(os.path.join(subdir, f"{save_idx}.json"), "w") as f:
        json.dump(arguments, f, indent=2)
    print(f"Saved arguments to {subdir}/{save_idx}.json for debugging.")


def save_txt_examples(save_txt_examples, reasoning_content=None, final_content=None, tool_calls=None, prefix="tmp"):
    save_idx = 0
    subdir = os.path.join("tmp_harmony", prefix)
    os.makedirs(subdir, exist_ok=True)
    while os.path.exists(os.path.join(subdir, f"{save_idx}.txt")):
        save_idx += 1
    with open(os.path.join(subdir, f"{save_idx}.txt"), "w") as f:
        f.write(save_txt_examples)
    if reasoning_content:
        with open(os.path.join(subdir, f"{save_idx}_reasoning.txt"), "w") as f:
            f.write(reasoning_content)
    if final_content:
        with open(os.path.join(subdir, f"{save_idx}_final.txt"), "w") as f:
            f.write(final_content)
    if tool_calls:
        with open(os.path.join(subdir, f"{save_idx}_tools.json"), "w") as f:
            json.dump(tool_calls, f, indent=2)
    print(f"Saved to {subdir}/{save_idx}.txt for debugging.")


            
def harmony_parse_calls(tokenizer, token_ids, txt, keep_tools: bool = True):
    # 1. Keep track of everything, but also split on first <|start|>
    token_ids_input = token_ids
    token_txt_input = tokenizer.decode(token_ids_input)
    txt_input = txt

    start_id = tokenizer.encode("<|start|>")[0]

    prefix_txt = ""
    if start_id in token_ids_input:
        start_pos = token_ids_input.index(start_id)
        prefix_ids = token_ids_input[:start_pos]  # what you used to throw away
        token_ids = token_ids_input[start_pos:]   # the structured part
        if prefix_ids:
            prefix_txt = tokenizer.decode(prefix_ids)
    else:
        # No structured header at all, treat everything as prefix
        prefix_txt = tokenizer.decode(token_ids_input)
        token_ids = []  # nothing left for Harmony parser

    # 2. Decode once for the structured part
    token_txt = tokenizer.decode(token_ids) if token_ids else ""
    
    # --- try to treat prefix as a Harmony message ---
    synthetic_prefix = sanitize_start_end_message(prefix_txt) if prefix_txt else ""
    token_txt = sanitize_start_end_message(token_txt) if token_txt else ""
    
    if synthetic_prefix:
        # Prepend synthetic message to the structured text
        # If wrap prefix failed: we do not trust the prefix and hence we would throw it away (instead of prepending it to reasoning_content) 
        token_txt = synthetic_prefix + token_txt
        # prefix_txt = ""

    # 3. Patch corner cases
    token_txt = sanitize_harmony_headers_keep_tools(token_txt)

    # 4. Re-encode after sanitization (if there is anything left)
    token_ids = tokenizer.encode(token_txt) if token_txt else []

    try:
        reasoning_content = ""
        final_content = ""
        tool_calls = [] 

        if token_ids:
            # Parse into Harmony messages
            parser = parse_output_into_messages(token_ids)
            messages = parser.messages

            # Use the new helpers to turn messages + remaining state into ResponseOutputItems
            output_items: list[ResponseOutputItem] = []
            for msg in messages:
                output_items.extend(parse_output_message(msg))
            
            # Capture any partially streamed content
            output_items.extend(parse_remaining_state(parser))

            # Accumulate:
            #  - raw reasoning chunks (reasoning items + non-final messages + fn-call args if keep_tools=False)
            #  - assistant message texts (for picking the last as final)
            #  - tool_calls (if keep_tools=True)
            reasoning_chunks: list[str] = []
            assistant_msg_texts: list[str] = []
            tool_calls: list[dict] = []

            for item in output_items:
                item_type = getattr(item, "type", None)

                # --- Reasoning items ---
                if item_type == "reasoning":
                    for c in getattr(item, "content", []):
                        if getattr(c, "type", None) == "reasoning_text":
                            text = getattr(c, "text", "") or ""
                            if text:
                                reasoning_chunks.append(text)

                # --- Assistant messages (collect all texts first) ---
                elif item_type == "message":
                    role = getattr(item, "role", None)
                    if role == "assistant":
                        texts: list[str] = []
                        for c in getattr(item, "content", []):
                            if getattr(c, "type", None) == "output_text":
                                t = getattr(c, "text", "") or ""
                                if t:
                                    texts.append(t)
                        if texts:
                            assistant_msg_texts.append("\n".join(texts))

                # --- Tool calls ---
                elif item_type == "function_call":
                    fn_name = getattr(item, "name", None)
                    arguments = getattr(item, "arguments", "") or ""
                    
                    is_code = fn_name in ["code", "python", "code_execution"] or "code" in fn_name.lower() or "python" in fn_name.lower()

                    # try strip arguments, some times to=functions code<|message|> xxx mess up parsing, parsed to code<|message|> xxx as the content
                    arguments, open_braces = extract_tool_args(arguments)
                    
                    parsed_args = {}
                    if open_braces == 0:
                        if is_code:
                            parsed_args = {"code": arguments} 
                        else: 
                            print("Empty function name!!!")
                            save_txt_examples(token_txt_input, prefix="tmp_args_no_code")
                            parsed_args = {"text": arguments}
                    else:
                        try:
                            parsed_args = json.loads(arguments)
                        except:
                            try:
                                parsed_args = json.loads(repair_json(arguments))
                                print("argument error (basic) but ok with json reapir!")
                                save_txt_examples(token_txt_input, prefix="tmp_args_json_error_ok_repair")
                                save_arguments(arguments, prefix="tmp_args_json_error_ok_repair")
                            except:
                                print("argument error (json repair)!!!")
                                save_txt_examples(token_txt_input, prefix="tmp_args_json_error_after_repair")
                                save_arguments(arguments, prefix="tmp_args_json_error_after_repair")
                    if keep_tools:
                        tool_calls.append({"name": fn_name, "arguments": arguments, 
                                           "role": "assistant", "channel": "commentary"})
                    else:
                        # Sometimes even if we tell it to output ```python ... ```, it still tries to do a python tool call. 
                        # Let's handle that case for keep_tools = False by mapping it back to ```python ... ``` syntax.             
                        # Otherwise, fold tool-call args into reasoning if we're not keeping tools
                        if parsed_args:
                            found_code = False
                            # edge case: no name
                            if fn_name in ["code", "python", "code_execution", ""] or "code" in fn_name.lower() or "python" in fn_name.lower():
                                for key in ["code", "python", "python_code"]:
                                    if isinstance(parsed_args, dict) and key in parsed_args:
                                        code_snippet = parsed_args[key]
                                        assistant_msg_texts.append(f"```python\n{code_snippet}\n```")
                                        found_code = True
                                        break
                            
                            if not found_code:
                                if isinstance(parsed_args, dict):
                                    for key in parsed_args:
                                        if key == "text":
                                            assistant_msg_texts.append(parsed_args[key])
                                        else:
                                            assistant_msg_texts.append(f"{key}: {parsed_args[key]}")
                                else:
                                    assistant_msg_texts.append(f"{parsed_args}")
                else:
                    raise ValueError(f"Unknown output item type: {item_type}")
         
            # Now split assistant messages into reasoning vs final
            last_final_text: str | None = None
            if assistant_msg_texts:
                # All but the last go into reasoning
                last_final_text = assistant_msg_texts[-1]
                reasoning_chunks.extend(assistant_msg_texts[:-1])
            
            reasoning_content = "\n".join(reasoning_chunks)
            final_content = last_final_text or ""
            
        # Debug: check if we lost a lot of content
        if len(reasoning_content) + len(final_content) + len(str(tool_calls)) + 200 < len(token_txt_input):
            diff_token = len(token_txt_input) - (len(reasoning_content) + len(final_content) - len(str(tool_calls)))
            print(f"Warning: parsed content is much ({diff_token} tokens) shorter than original token_txt_input.")
            save_txt_examples(token_txt_input, reasoning_content, final_content, tool_calls, prefix="tmp")

        if keep_tools:
            return reasoning_content, final_content, tool_calls
        else:
            return reasoning_content, final_content

    except Exception as e:
        print(f"Error **parse_output_into_messages**: {str(e)[:200]}")
        save_txt_examples(token_txt_input, prefix="error")
        # On error, at least return the original txt as "final" + prefix as reasoning
        if keep_tools:
            return "", txt, []
        else:
            return "", txt
#############################################################    
