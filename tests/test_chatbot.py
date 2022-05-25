from Companion.Gpt3.gpt3 import GPT3, create_chat_prompt

import datetime
start = datetime.datetime.now()

user_name = "Logan"

ai_name = "Nick"
examples = None

while True:
    human_input = f"\n{user_name}: {input()}"

    new_examples, start_prompt = create_chat_prompt(user_name, ai_name, examples)

    prompt = start_prompt + human_input

    chat = GPT3(prompt=prompt, human_input=human_input, start_text=f"\nPositive {ai_name}:",
                restart_text=f'\n {user_name}:', api_key="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")

    AI, new_examples = chat.basic_chat(new_examples)

    examples = new_examples

    print(AI, "\n")

    finish = datetime.datetime.now()
    print(finish-start)