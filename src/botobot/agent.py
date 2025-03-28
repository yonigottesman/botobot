from typing import Callable, get_type_hints


class Tool:
    def __init__(self, function: Callable, call_args={}):
        if function.__doc__ is None:
            raise ValueError("Tool functions must have a docstring describing the tool")

        self.function = function
        self.description = function.__doc__
        self.name = function.__name__
        if "inputs" not in get_type_hints(function):
            self.input_model = None
            self.input_schema = {"type": "object", "properties": {}, "required": []}
        else:
            self.input_model = get_type_hints(function)["inputs"]
            self.input_schema = self.input_model.model_json_schema()

        self.additional_args = call_args


class ToolsContainer:
    def __init__(self, tools: list[Tool]):
        self.tooldict = {t.name: t for t in tools}

    def run_tool(self, tool_name, inputs):
        try:
            tool_instance = self.tooldict[tool_name]
            inputs = tool_instance.input_model(**inputs) if tool_instance.input_model is not None else None
            if inputs is None:
                result = tool_instance.function(**tool_instance.additional_args)
            else:
                result = tool_instance.function(inputs=inputs, **tool_instance.additional_args)
        except Exception as e:
            result = e
        return "Tool Result:\n" + str(result)

    def claude_format(self):
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tooldict.values()
        ]


def drop_citations_bug(messages):
    new_messages = []
    for m in messages:
        if m["role"] == "user":
            new_messages.append(m)
        elif m["role"] == "assistant":
            new_content = []
            for c in m["content"]:
                new_c = dict(**c)
                new_c.pop("citations", None)
                new_content.append(new_c)
            m["content"] = new_content
            new_messages.append(m)
    return new_messages


def agentic_steps(messages, claude_client, tools: ToolsContainer, system_prompt, callback, model):
    while True:
        response = claude_client.messages.create(
            model=model,
            max_tokens=8192,
            tools=tools.claude_format(),
            system=system_prompt,
            messages=drop_citations_bug(messages),
            temperature=0.0,
        )
        response_message = {"role": "assistant", "content": [c.model_dump() for c in response.content]}
        messages.append(response_message)

        if response.stop_reason == "tool_use":
            callback(response_message)
            for content in response.content:
                if content.type == "tool_use":
                    tool_result = tools.run_tool(content.name, content.input)
                    new_message = {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": tool_result,
                            }
                        ],
                    }
                    callback(new_message)
                    messages.append(new_message)

        else:
            return response_message
