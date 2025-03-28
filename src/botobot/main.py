import io
import os
import re
from contextlib import redirect_stdout
from datetime import datetime
from functools import partial
from typing import Annotated

import boto3
import typer
import yaml
from anthropic import AnthropicBedrock
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from pydantic import BaseModel, Field
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

from botobot.agent import Tool, ToolsContainer, agentic_steps

HISTORY_FILE = os.path.expanduser("~/.botobot/cache")


def setup_history():
    history_dir = os.path.dirname(HISTORY_FILE)
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)


app = typer.Typer(help="Boto3 Agent application")
console = Console()


def get_available_boto3_services():
    """Get all available services from the AWS account. These are services that can be used with boto3.client"""
    return yaml.dump(boto3.session.Session().get_available_services())


def get_available_profiles():
    """Get all available profiles from the AWS account. These are profiles that can be used with boto3.Session"""
    home = os.path.expanduser("~")
    if os.path.exists(os.path.join(home, ".aws", "config")):
        with open(os.path.join(home, ".aws", "config"), "r") as f:
            config = f.read()
        profiles = re.findall(r"\[profile (.*)\]", config)
        return yaml.dump(profiles)
    else:
        return "No profiles found in ~/.aws/config. Using default aws credentials"


class Boto3Command(BaseModel):
    service: str = Field(..., description="The service to use. will be called as boto3.client(service)")
    action: str = Field(..., description="The action to use. will be called as client.action()")
    parameters: dict = Field(..., description="The parameters to use. will be passed to the action as kwargs")
    region: str = Field(..., description="The region to use. will be passed to the action as kwargs")
    profile_name: str | None = Field(
        None,
        description="Will be passed to the boto3.Session as profile_name. Keep empty if you want to use the default aws credentials",
    )


def get_boto3_command_documentation(service, action):
    session = boto3.Session()
    boto3_client = session.client(service)
    action = getattr(boto3_client, action)

    f = io.StringIO()
    with redirect_stdout(f):
        help(action)

    help_text = f.getvalue()

    return help_text[: help_text.find(":rtype:")]


def run_boto3_command(inputs: Boto3Command):
    """Run a boto3 command and return the result"""
    session = boto3.Session(region_name=inputs.region, profile_name=inputs.profile_name)
    boto3_client = session.client(inputs.service)
    result = getattr(boto3_client, inputs.action)(**inputs.parameters)
    return yaml.dump(result)


class Boto3CommandRequest(BaseModel):
    service: str = Field(..., description="The service to use. will be called as boto3.client(service)")
    action: str = Field(..., description="The action to use. will be called as client.action()")
    profile_name: str = Field(
        ...,
        description="Will be passed to the boto3.Session as profile_name. Keep empty if you want to use the default aws credentials",
    )
    region: str = Field(..., description="Will be passed to the boto3.Session as region_name")
    boto_request: str = Field(
        ...,
        description="Detailed description of the action to perform and the output you expect. This text will be used to generate the parameters for the action.",
    )
    required_learnings: str = Field(
        ...,
        description="After the boto3 command is executed, these are the learnings that should be extracted from the boto3 result. Use the learnings to filter the result as much as possible",
    )


def run_boto3_command_request(inputs: Boto3CommandRequest, bedrock_client, bedrock_model_name):
    """Generate and run a boto3 command. Return the required learnings. Use the learnings and description to filter as much as possible.
    If there is a nextToken it will be returned as a learning. so that it can be used for pagination. call the function again with it to iterate through the results
    """

    prompt = f"""
    You are a helpful assistant that can access the AWS API using boto3. today is {datetime.now().strftime("%Y-%m-%d")}.
    You are given a description of the action to perform and the documentation for the action.
    Your job is to generate the parameters for the action.
    The action is: {inputs.action}
    The service is: {inputs.service}
    The documentation for the action is:
    {get_boto3_command_documentation(inputs.service, inputs.action)}
    The description of the action to perform is:
    {inputs.boto_request}
    The region to use is: {inputs.region}
    The profile to use is: {inputs.profile_name}
    """

    response = bedrock_client.messages.create(
        model=bedrock_model_name,
        max_tokens=4096,
        tools=[
            {
                "name": "run_boto3_command",
                "description": "Run a boto3 command and return the result",
                "input_schema": Boto3Command.model_json_schema(),
            },
        ],
        tool_choice={"type": "tool", "name": "run_boto3_command"},
        system=system_prompt,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=0.0,
    )
    boto_command = Boto3Command(**response.content[0].input)
    result = run_boto3_command(boto_command)

    learning_prompt = f"""
    Given a boto3 request description and the result of the request, extract the relevant information from the result.
    Be concise, return only the learnings you are requested. nothing more.
    IF there is a nextToken in the result, return it as a learning. so that it can be used for pagination.
    The required learnings are:
    {inputs.required_learnings}
    The request description is:
    {inputs.boto_request}
    The Command is:
    {inputs.action}
    The result of the request is:
    {result}
    """
    response = bedrock_client.messages.create(
        model=bedrock_model_name,
        max_tokens=8192,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": learning_prompt}]},
        ],
        temperature=0.0,
    )
    response = f"boto call: {boto_command}\n\nlearnings: {response.content[0].text}"
    return response


def get_tools(bedrock_client, bedrock_model_name):
    tools = [
        Tool(get_available_boto3_services),
        Tool(
            run_boto3_command_request,
            call_args={
                "bedrock_client": bedrock_client,
                "bedrock_model_name": bedrock_model_name,
            },
        ),
        Tool(get_available_profiles),
    ]
    return ToolsContainer(tools)


def display_assistant_substep(message):
    for content in message["content"]:
        if content["type"] == "text":
            console.print(Panel(content["text"], title="Reasoning", border_style="green"))
        elif content["type"] == "tool_use":
            text = Markdown(f"Using tool: **{content['name']}** with input:")
            inputs = Syntax(yaml.dump(content["input"]), "yaml", theme="monokai", line_numbers=False)
            console.print(Panel(Group(text, inputs), title="Tool Use", border_style="dim green", style="dim"))
        elif content["type"] == "tool_result":
            console.print(
                Panel(
                    content["content"][:2000] + ("\n..." if len(content["content"]) > 2000 else ""),
                    title="Tool Result",
                    border_style="dim green",
                    style="dim",
                )
            )


system_prompt = f"""
You are a helpful assistant that can access the AWS API using boto3. today is {datetime.now().strftime("%Y-%m-%d")}.
Always try to filter results using the parameters in the commands.
Always present the user with the available profiles and ask for confirmation.
The user cannot see the tool results. Always repeat the tool results in your response. 
The user will only see the final response, not the resoning steps, so include all the information in the final response.
Tools are expensive. If you have a result from a previous tool use, use it to answer the user's question.
Be concise with your responses, dont get any more information than the user asks for.
"""


@app.command()
def chat(
    bedrock_profile_name: Annotated[str, typer.Option(help="The name of the bedrock profile to use")] = None,
    bedrock_model_name: Annotated[
        str, typer.Option(help="The name of the bedrock model to use")
    ] = "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    bedrock_region: Annotated[str, typer.Option(help="The region of the bedrock model to use")] = "us-east-1",
):
    messages = []
    setup_history()

    bedrock_client = AnthropicBedrock(aws_region=bedrock_region, aws_profile=bedrock_profile_name)

    tools = get_tools(bedrock_client, bedrock_model_name)

    session = PromptSession(history=FileHistory(HISTORY_FILE))

    console.print("[bold green]🤖[/]")

    while True:
        user_input = session.prompt("> ")
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": [{"type": "text", "text": user_input}]})
        with console.status("[bold green]Thinking..."):
            answer = agentic_steps(
                messages=messages,
                claude_client=bedrock_client,
                tools=tools,
                system_prompt=system_prompt,
                callback=partial(display_assistant_substep),
                model=bedrock_model_name,
            )

        console.print(Panel(Markdown(answer["content"][0]["text"]), title="Final Response", border_style="green"))


if __name__ == "__main__":
    app()
