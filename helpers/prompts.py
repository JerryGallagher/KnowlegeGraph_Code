import sys
from yachalk import chalk
sys.path.append("..")

import json
import ollama.client as client


def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[data, ui, class, package, controller, service, interface, factory, prototype, adapter, template, state, object, composite, interpreter, builder, facade, misc]\n"
        "Ignore low level datatypes, this.objects, constructors, and destructors "
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
        
    return result

def classify_topic(input_data: dict, model="mistral-openorca:latest"):
    if model is None:
        model = "mistral-openorca:latest"

    # You can fetch model info here if needed
    # model_info = client.show(model_name=model)
    # print(chalk.blue(model_info))

    SYS_PROMPT = (
        "You are a system for classifying groups of terms into topic categories. "
        "For each group of terms given, determine a general topic name that best describes the group. "
        "Consider the terms’ meanings, applications, and relationships with each other.\n\n"
        "Your response format should be a JSON list where each entry has:\n"
        "- 'group': The original terms in the group\n"
        "- 'topic': The identified general topic name\n\n"
        "Example:\n"
        '[{"group": ["term1", "term2"], "topic": "Relevant Topic"}]'
    )

    # Format the input data into a suitable prompt
    group_prompts = "\n\n".join(
        f"Group {i+1}: {', '.join(group)}" for i, group in enumerate(input_data["Communities"])
    )
    USER_PROMPT = f"context: ```{group_prompts}``` \n\n output: "

    response = openai.Completion.create(
        model=model,
        prompt=SYS_PROMPT + USER_PROMPT,
        max_tokens=500,
        temperature=0.3
    )

    try:
        result = json.loads(response.choices[0].text)
    except json.JSONDecodeError:
        print("\n\nERROR ### Here is the buggy response: ", response.choices[0].text, "\n\n")
        result = None
    
    return result

def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
        "You are a network graph maker who extracts terms and their relations from a given context. "
        "You are provided with a context chunk (delimited by ```) Your task is to extract the ontology "
        "of terms mentioned in the given context. These terms should represent the key concepts as per the context. \n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Think about how these terms can have one on one relation with other terms.\n"
            "\tTerms that are mentioned in the same sentence or the same paragraph are typically related to each other.\n"
            "\tTerms can be related to many other terms\n\n"
        "Thought 3: Find out the relation between each such related pair of terms. \n\n"
        "Format your output as a list of json. Each element of the list contains a pair of terms"
        "and the relation between them, like the follwing: \n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, {...}\n"
        "]"
    )
    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    
    return result
