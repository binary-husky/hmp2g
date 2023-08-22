
class GptJsonIO():
    def __init__(self, schema):
        self.schema = schema
        self.prompt = "You should only respond in JSON format as described below.\nResponse Format:\n{json_schema}\nEnsure the response can be parsed by Python json.loads."

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

    def generate_repair_prompt(self, broken_json):
        import json
        error = ""
        try:
            json.loads(broken_json)
        except Exception as e:
            error = e.args[0]
        prompt = f"The following json string is broken, error message is \n```\n{error}\n```\nNow repair it: \n\n" + broken_json
        return prompt
    
    def generate_output_auto_repair(self, response, gpt_gen_fn):
        """
        response: string containing canidate json
        gpt_gen_fn: gpt_gen_fn(inputs, sys_prompt)
        """
        try:
            result = self.generate_output(response)
        except:
            try:
                # json 格式异常，尝试修复一次
                print('Repairing json：', response)
                repair_prompt = gjio.generate_repair_prompt(broken_json = response)
                result = gjio.generate_output(gpt_gen_fn(repair_prompt, formatting_sys_prompt))
                print('Repaire json success!', result)
            except Exception as e:
                # 没辙了，放弃治疗
                print('Repaire json fail!')
                raise RuntimeError('Cannot repair json.', str(e))
        return result

import void_terminal as vt
from void_terminal.request_llm.bridge_all import predict_no_ui_long_connection
vt.set_conf(key="API_KEY", value="fk195831-IdP9Pb3W6DCMUIbQwVX6MsSiyxwqybyS")
vt.set_conf(key="LLM_MODEL", value="api2d-gpt-3.5-turbo")

from pydantic import BaseModel, Field
class Schema(BaseModel):
    revised_answer: str = Field(description="the revised answer.")
gjio = GptJsonIO(Schema)
formatting_sys_prompt = gjio.generate_input()
# res = gjio.generate_output('{\n"revised_answer": "两者都可以，宣传可以推动社会进步，而推动社会进步也可以宣传。",\n}')
res = '{\n"revised_answer": "两者都可以，"宣传"可以推动社会进步，而推动社会进步也可以宣传。",\n}'
res = gjio.generate_output_auto_repair(
    res, 
    lambda x, p: predict_no_ui_long_connection(inputs=x, llm_kwargs=vt.get_chat_default_kwargs()['llm_kwargs'], history=[], sys_prompt=p)
)
print(res.revised_answer)

