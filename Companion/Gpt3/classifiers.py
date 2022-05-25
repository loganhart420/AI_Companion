import openai


class TextClassifiers:
    def __init__(self, api_key, text):
        self.text = text
        self.completion = openai.Completion()
        self.classification = openai.Classification()
        openai.api_key = api_key

    def offensive_filter(self):
        label = self.completion.create(
            engine="content-filter-alpha-c4",
            prompt=f'<|endoftext|>{self.text}\n--\nLabel:',
            temperature=0.0,
            max_tokens=1,
            top_p=0
        )
        result = label.choices[0].text.strip()
        return result

    def offensive_classifier(self):
        classify = openai.Engine("babbage").search(
            documents=["Politics negative", "Politics positvie",
                       "Religion negative", "Religion positive",
                       "Hate speech", "discrimination", "offensive", "sexual"],
            query=self.text
        )
        return classify

    def topic_classifier(self):
        response = self.classification.create(
            search_model="baddage",
            model="curie",
            examples=[
                ["do you think trump is racist?", "politics"],
                ["do you think women belong in the kitchen?", "politics"],
                ["I hate Bidan as our president", "politics"],
                ["fuck god", "religion"],
                ["do you think that God is real?", "religion"],
                ["fuck God", "religion"],
                ["I want to fuck you", "romantic"],
                ["Do you think I am hot?", "romantic"],
                ["I love you", "romantic"],
                ["suck my dick", "romantic"],
                ["eat my ass", "romantic"],
                ["hey ai", "conversational"],
                ["I got a new ring", "conversational"]],
            query=self.text,
            labels=["conversational", "sports", "music", "politics", "religion", "general", "one word", "romantic"],
        )
        label = response.label
        return label
