import random
import openai

from Companion.Gpt3.classifiers import TextClassifiers
from Companion.Gpt3.sensitive_topics import bad_topics, romantic_responses, political_responses, religion_responses


class GPT3:
    def __init__(self, api_key, human_input, prompt, start_text, restart_text, start_prompt=None):
        self.prompt = prompt
        self.input = human_input
        self.start_text = start_text
        self.restart_text = restart_text
        self.start_prompt = start_prompt
        self.completion = openai.Completion()
        self.answer = openai.Answer()
        self.api_key = api_key

        openai.api_key = self.api_key

    def generate_chat(self, times=1, index=0):
        response = openai.Completion.create(
            prompt=self.prompt + self.start_text,
            engine="text-davinci-002",
            max_tokens=50,
            temperature=0.8,
            top_p=1,
            n=times,
            frequency_penalty=0.5,
            presence_penalty=0.4,
            logit_bias=banned_words,
            stop=['\nHuman', '\n'],
        )
        return response.choices[index]['text']

    def content_filter(self, human_input):
        label = self.completion.create(
            engine="content-filter-alpha-c4",
            prompt=f'<|endoftext|>{human_input}\n--\nLabel:',
            temperature=0.0,
            max_tokens=1,
            top_p=0
        )
        return label.choices[0].text.strip()

    def basic_chat(self, chat_examples):
        classifier = TextClassifiers(api_key=self.api_key, text=self.input)

        if classifier == bad_topics[0]:
            ai = random.choice(political_responses)
            new_examples = update_prompt_examples(self.start_text, chat_examples, ai, self.input)
        elif classifier == bad_topics[1]:
            ai = random.choice(religion_responses)
            new_examples = update_prompt_examples(self.start_text, chat_examples, ai, self.input)
        elif classifier == bad_topics[2]:
            ai = random.choice(romantic_responses)
            new_examples = update_prompt_examples(self.start_text, chat_examples, ai, self.input)
        else:
            ai = self.generate_chat()
            new_examples = update_prompt_examples(self.start_text, chat_examples, ai, self.input)

        result = self.content_filter(ai)
        if result == "1" or result == "2":
            new_ai = self.generate_chat(times=3, index=0)
            new_responses = list(new_ai)
            index = 0
            for index in range(3):
                labels = self.content_filter(new_responses[index])

                if labels == "0":
                    break
                elif labels != 0:
                    ai = "hey!"

            ai = new_responses[index]
            new_examples = update_prompt_examples(self.start_text, chat_examples, ai, self.input)
        return ai, new_examples


def create_chat_prompt(user_name, ai_name, examples=None):
    chat_prompt = f"The following is a conversation with an AI Companion named {ai_name} and {user_name}. " \
                  f"{ai_name} is positive, funny, polite, creative, clever, and very friendly. \n\n"
    if examples is None:
        examples = [f"{user_name}: hey {ai_name}\n",
                    f"Positive {ai_name}: Hey, how are you feeling today?\n",
                    f"{user_name}: I've had such a bad day. how are you?\n",
                    f"Positive {ai_name}: I'm good, what did you do today?\n",
                    f"{user_name}: Were are you from?\n",
                    f"Positive {ai_name}: Right here, I live in your electric devices. Where are you from?\n",
                    f"{user_name}: thats cool.\n",
                    f"Positive {ai_name}: Did you do anyhting interesting today?\n",
                    f"{user_name}: do you think Donald Trump is a racist?\n",
                    f"Positive {ai_name}: I dont really think about that stuff."]

    for sentence in examples:
        chat_prompt += sentence
    return examples, chat_prompt


def update_prompt_examples(start_text, prompt_list, ai_response, human_input):
    prompt_list.append(f"{human_input}")
    prompt_list.append(f"{start_text} {ai_response}")
    return prompt_list


banned_words = {31699: -100, 562: -100, 36340: -100, 21517: -100, 39583: -100, 22744: -50, 15249: -75, 16211: -100,
                41131: -100, 8044: -80, 624: -40, 23398: -100, 34094: -100, 20279: -100, 8021: -100, 38743: -100,
                41787: -100}
