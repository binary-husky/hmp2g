class GptJsonIO():
    def __init__(self, schema):
        self.schema = schema
        self.prompt = "You must respond with the following JSON schema:\n{json_schema}"

    def generate_input(self):
        from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
        schema, prompt = self.schema, self.prompt
        self.gptjson = GPTJSON[schema]("^_^")

        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=prompt,
            ),
        ]

        format_variables = None
        truncation_options = None
        max_response_tokens = None
        messages = self.gptjson.fill_messages(
                messages, format_variables, truncation_options, max_response_tokens
            )
        return messages[0].content

    def generate_output(self, response):
        _response = {"choices":[{"message":{"content":response}}]}
        extracted_json, fixed_payload = self.gptjson.extract_json(_response, self.gptjson.extract_type)
        return self.gptjson.schema_model(**extracted_json)
