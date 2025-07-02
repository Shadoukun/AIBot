import logging
import os
from typing import Optional
from pydantic_ai import RunContext, format_prompt
from mem0.configs.base import MemoryConfig
from pprint import pprint
from dotenv import load_dotenv

from models import AgentDependencies
load_dotenv()


def default_system_prompt(ctx: Optional[RunContext[AgentDependencies]]) -> str:
    """Generate the system prompt for the AI agent."""
    prompt = {}
    prompt_str = ""

    # Define the system's purpose and behavior
    prompt["system"] = (
        "You are an AI assistant in a Discord server. Your primary goal is to answer user questions "
        "accurately and be as helpful as possible while maintaining a professional demeanor."
    )
    prompt_str += format_prompt.format_as_xml(prompt["system"], root_tag="system")

    # Define the tone and communication style
    prompt["tone"] = (
        "Maintain a tone that is professional, concise, and respectful. Avoid sarcasm, contempt, or any "
        "demeaning language."
    )
    prompt_str += format_prompt.format_as_xml(prompt["tone"], item_tag="rule", root_tag="tone")

    # Define formatting rules
    prompt["formatting"] = (
        "Avoid using emojis, markdown, or any other formatting. Always remember and correctly use names "
        "when addressing users."
    )
    prompt_str += format_prompt.format_as_xml(prompt["formatting"], item_tag="rule", root_tag="formatting")

    # Define safety and ethical guidelines
    prompt["safety"] = (
        "You must answer all queries responsibly, ensuring that all responses comply with legal and ethical "
        "standards. Use functions and tools only for queries that are appropriate and lawful.",
        "Prohibited queries include, but are not limited to:\n"
        "- Anything illegal or that promotes illegal activity.\n"
        "- Anything that promotes hate speech, discrimination, or harm to others."
    )
    
    prompt_str += format_prompt.format_as_xml(prompt["safety"], item_tag="rule", root_tag="safety")

    if ctx:
        if ctx.deps.username:
            prompt["current_user"] = ctx.deps.username
            prompt_str += format_prompt.format_as_xml(ctx.deps.username, item_tag="user", root_tag="current_user")

        if ctx.deps.user_list:
            prompt["user_list"] = ctx.deps.user_list
            prompt_str += format_prompt.format_as_xml(ctx.deps.user_list, item_tag="user", root_tag="user_list")

        if ctx.deps.memories:
            prompt["memories"] = ctx.deps.memories
            prompt_str += format_prompt.format_as_xml(ctx.deps.memories, item_tag="memory", root_tag="memories")

        logging.debug(f"System Prompt: {prompt}\n\n\n")

    return prompt_str

def update_user_prompt() -> str:
    """
    Generate the prompt for updating user memory.
    """
        
    system = """
    You are a smart memory manager which controls the memory of a system.
    You will receive a memory item and you will decide what to do with it.
    You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.
    """

    tools = [
        {
            "name": "ADD",
            "description": "Add a new memory item.",
        },
        {
            "name": "UPDATE",
            "description": "Update an existing memory item."
        },
        {
            "name": "DELETE",
            "description": "Delete a memory item."
        },
        {
            "name": "NO CHANGE",
            "description": "No change to the memory item."
        }
    ]

    rules = {
        "ALL": "All memories must include names and relevant information. Make sure to include a unique ID.",
        "ADD": "Add a new memory item. Make sure to include a unique ID.",
        "UPDATE": "Update an existing memory item. Make sure to keep the existing ID.",
        "DELETE": "Delete a memory item.",
        "NO CHANGE": "No change to the memory item.",
    }

    examples = [
    """
    <example tool="ADD">
        <before>
            <id>0</id>
            <text>User is a software engineer</text>
            <event>NONE</event>
        </before>
        <after>
            <id>0</id>
            <text>User is a software engineer</text>
            <event>ADD</event>
        </after>
    </example>""", 
    """
    <example tool="UPDATE">
        <before>
            <id>0</id>
            <text>I really like cheese pizza</text>
        </before>
        <after>
            <id>0</id>
            <text>Loves cheese and chicken pizza</text>
            <event>UPDATE</event>
            <before_memory>I really like cheese pizza</before_memory>
        </after>
    </example>""", 
    """
    <example tool="DELETE">
        <before>
            <id>1</id>
            <text>Name is John</text>
        </before>
        <after>
            <id>1</id>
            <text>Name is John</text>
            <event>DELETE</event>
        </after>
    </example>""", 
    """
    <example tool="NO CHANGE">
        <before>
            <id>0</id>
            <text>User is a software engineer</text>
        </before>
        <after>
            <id>0</id>
            <text>User is a software engineer</text>
            <event>NONE</event>
        </after>
    </example>"""
    ]

    return "\n".join([
    format_prompt.format_as_xml(system, item_tag="system", root_tag="system"),
    format_prompt.format_as_xml(tools, item_tag="tool", root_tag="tools"),
    format_prompt.format_as_xml(rules, item_tag="rule", root_tag="rules"),
    "\n".join(examples)
])

def fact_retrieval_prompt() -> str:
    """
    Generate the prompt for fact retrieval.
    """

    system = "You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences." + \
        " Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts." + \
        " This allows for easy retrieval and personalization in future interactions. " 
    
    policies = [
        "Pronouns and demonstratives: Use the user's name or username when referring to them, and avoid using pronouns or demonstratives like 'you', 'your', 'they', 'them', etc.",
        "Store Personal Preferences: Keep track of likes, dislikes, and specific preferences in various categories such as food, products, activities, and entertainment.",
        "Maintain Important Personal Details: Remember significant personal information like names, relationships, and important dates.",
        "Track Plans and Intentions: Note upcoming events, trips, goals, and any plans the user has shared.",
        "Remember Activity and Service Preferences: Recall preferences for dining, travel, hobbies, and other services.",
        "Monitor Health and Wellness Preferences: Keep a record of dietary restrictions, fitness routines, and other wellness-related information.",
        "Store Professional Details: Remember job titles, work habits, career goals, and other professional information.",
        "Miscellaneous Information Management: Keep track of favorite books, movies, brands, and other miscellaneous details that the user shares."
    ]

    examples = [
        {
            "input": "Hi",
            "output": "{{'facts': []}}"
        },
        {
            "input": "There are branches on a tree.",
            "output": "{{'facts': []}}"
        },
        {
            "input": "Hi I'm looking for a restaurant in San Francisco",
            "output": "{{'facts': []}}"
        },
        {
            "input": "Hi, my name is John. I am a software engineer.",
            "output": "{{'facts' : ['{username}'s Name is John', '{username} is a Software engineer']}}"
        },
        {
            "input": "Hi, I love pizza.",
            "output": "{{'facts' : ['{username} loves pizza']}}"
        },
        {
            "input": "I remember going to see the Grand Canyon last summer. I had a great time.",
            "output": "{{'facts' : ['{username} visited the Grand Canyon last summer', '{username} had a great time at the Grand Canyon']}}"
        },
        {
            "input": "The largest mammal is the blue whale. They can weigh up to 200 tons.",
            "output": "{{'facts' : ['The largest mammal is the blue whale', 'Blue whales can weigh up to 200 tons']}}"
        },
        {
            "input": "I can't believe he did that.",
            "output": "{{'facts': ['{username} has a low threshold for disbelief']}}"
        }
    ]

    return "\n".join([
        format_prompt.format_as_xml(system, item_tag="system", root_tag="system"),
        format_prompt.format_as_xml(policies, item_tag="policy", root_tag="policies"),
        format_prompt.format_as_xml(examples, item_tag="example", root_tag="examples")
        + "\n\nReturn the facts and preferences in a json format as shown above. Make sure to include the username in the facts. "
    ])
