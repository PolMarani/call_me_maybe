from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model
from typing import Any
import numpy as np
import textwrap
import json
import re


class ConstrainedDecoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Small_LLM_Model
    functions: list
    prefix_structure: dict = {}
    function_name_tokens: list = []
    attributes_names_tokens: dict = {}
    number_tokens: list = []
    string_tokens: list = []
    comma_token: int = 0
    close_brace_token: int = 0

    def model_post_init(self, __context: Any) -> None:
        self.prefix_structure = {
            "START": self.model.encode('{').tolist()[0],
            "NAME_KEY": self.model.encode('"name": "').tolist()[0],
            "FINISHED_FUNCTION": self.model.encode('"').tolist()[0],
            "PARAMS_OPEN": self.model.encode(', "parameters": {"').tolist()[0],
            "VALUE_DEFINITION_STRING": self.model.encode('": "').tolist()[0],
            "VALUE_DEFINITION_NUMBER": self.model.encode('": ').tolist()[0],
            "MULTIPLE_PARAM_STRING": self.model.encode(', "').tolist()[0],
            "MULTIPLE_PARAM_NUMBER": self.model.encode(' "').tolist()[0],
            "END_STRING": self.model.encode('}}').tolist()[0],
            "END_NUMBER": self.model.encode('}').tolist()[0]}

        self.function_name_tokens = [
            self.model.encode(function["name"]).tolist()[0]
            for function in self.functions]
        # Crea dict di liste, che sarebbero gli attributi per ogni funzione
        self.attributes_names_tokens = {
            function["name"]:
            [self.model.encode(param_name).tolist()[0]
             for param_name in function["parameters"].keys()]
            for function in self.functions
        }
        with open(self.model.get_path_to_vocab_file(), 'r') as f:
            vocab = json.load(f)
        self.number_tokens = [
            token_id for token_str, token_id in vocab.items()
            if all(c in "0123456789.-eE" for c in token_str)
        ]
        self.string_tokens = [
            token_id for token_str, token_id in vocab.items()
            if '"' not in token_str
        ]
        self.comma_token = self.model.encode(',').tolist()[0][0]
        self.close_brace_token = self.model.encode('}').tolist()[0][0]

        self.number_tokens.extend(
            [self.comma_token])
        self.number_tokens.extend(
            [self.close_brace_token])
        self.string_tokens.extend(
                    self.prefix_structure["FINISHED_FUNCTION"])

    def fix_regex_pattern(self, par_value: str) -> str:
        stripped = par_value.rstrip()
        if stripped.endswith('|'):
            stripped = stripped.rstrip('|')
        if stripped.endswith('.*'):
            m = re.fullmatch(r'([A-Za-z0-9_]+)\.\*', stripped)
            if m:
                stripped = m.group(1)
        return stripped

    def generate_function_call(self, prompt: str) -> dict:
        instructions = f"""
        You are a function calling assistant. Given a user request and a list
        of available functions, output a JSON object with the name of
        the function to call and its arguments.

        Available functions:
        {json.dumps(self.functions)}

        User request: {prompt}

        Output:
        """
        instructions = textwrap.dedent(instructions)
        input_ids = self.model.encode(instructions).tolist()[0]
        compatible_function = self.function_name_tokens.copy()
        gen_cont = 0
        attr_cont = 0
        state = "START"
        chosen_function = None
        prompt_len = len(input_ids)
        result = {}
        attr_type = ""
        value_cont = 0
        value_rep = [2, 3, 4, 5, 6, 7, 11]

        print(" - The prompt is:", prompt)
        print()
        while True:

            # state START
            if state == "START":
                input_ids += self.prefix_structure["START"]
                state = "NAME_KEY"
                continue
            # state NAME_KEY
            elif state == "NAME_KEY":
                input_ids += self.prefix_structure["NAME_KEY"]
                state = "FUNCTION_NAME"
                continue
            # state FUNCTION_NAME
            elif state == "FUNCTION_NAME":
                logits = self.model.get_logits_from_input_ids(input_ids)
                if (gen_cont == len(compatible_function[0])):
                    valid_tokens = self.prefix_structure["FINISHED_FUNCTION"]
                    chosen_function = self.model.decode(compatible_function[0])
                else:
                    valid_tokens = [function[gen_cont]
                                    for function in compatible_function]
            # state PARAMS_OPEN
            elif state == "PARAMS_OPEN":
                input_ids += self.prefix_structure["PARAMS_OPEN"]
                state = "PARAM_NAME"
                continue
            elif state == "PARAM_NAME":
                logits = self.model.get_logits_from_input_ids(input_ids)
                function = next(f for f in self.functions
                                if f["name"] == chosen_function)
                print(f"function found: \n {function}")
                print(f"parameters: \n {function['parameters']}")
                attribute = list(function["parameters"].keys())[attr_cont]
                print(f"PARAM_NAME: attr_cont={attr_cont}, "
                      f"chosen={chosen_function}")
                input_ids += (
                    self.attributes_names_tokens[chosen_function][attr_cont])
                attr_type = function["parameters"][attribute]["type"]
                attr_cont += 1
                if attr_type == "string":
                    state = "VALUE_DEFINITION_STRING"
                    continue
                elif attr_type == "number":
                    state = "VALUE_DEFINITION_NUMBER"
                    continue
            elif state == "VALUE_DEFINITION_STRING":
                input_ids += self.prefix_structure["VALUE_DEFINITION_STRING"]
                state = "PARAM_VALUE_STRING"
                continue
            elif state == "VALUE_DEFINITION_NUMBER":
                input_ids += self.prefix_structure["VALUE_DEFINITION_NUMBER"]
                state = "PARAM_VALUE_NUMBER"
                continue
            elif state == "PARAM_VALUE_STRING":
                logits = self.model.get_logits_from_input_ids(input_ids)
                if (value_cont >= 30):
                    valid_tokens = self.model.encode('"').tolist()[0]
                else:
                    valid_tokens = self.string_tokens
            elif state == "PARAM_VALUE_NUMBER":
                logits = self.model.get_logits_from_input_ids(input_ids)
                function = next(f for f in self.functions
                                if f["name"] == chosen_function)
                tot_attr = len(function["parameters"])
                if (value_cont >= 10) and attr_cont == tot_attr:
                    valid_tokens = [self.close_brace_token]
                elif (value_cont >= 10) and attr_cont != tot_attr:
                    valid_tokens = [self.comma_token]
                else:
                    if attr_cont == tot_attr:
                        number_tokens_nocomma = [
                                    t for t in self.number_tokens
                                    if t != self.comma_token
                                ]
                        valid_tokens = number_tokens_nocomma
                    else:
                        valid_tokens = self.number_tokens
            elif state == "MULTIPLE_PARAM":
                if attr_type == "string":
                    input_ids += self.prefix_structure["MULTIPLE_PARAM_STRING"]
                elif attr_type == "number":
                    input_ids += self.prefix_structure["MULTIPLE_PARAM_NUMBER"]
                state = "PARAM_NAME"
                continue
            elif state in ("END_STRING", "END_NUMBER"):
                input_ids += self.prefix_structure[state]
                final_json = input_ids[prompt_len:]
                print(repr(self.model.decode(final_json)))
                final_json = json.loads(self.model.decode(final_json))
                result["prompt"] = prompt
                result.update(final_json)
                if "regex" in result["parameters"]:
                    result["parameters"]["regex"] = (
                        self.fix_regex_pattern(result["parameters"]["regex"]))
                print(" ->")
                print("RESULT IS:", result)
                print()
                print("--------------------------------------------",
                      "------------")
                return result

            logits_array = np.array(logits)
            filtered_logits = np.full_like(logits_array, -np.inf)
            filtered_logits[valid_tokens] = logits_array[valid_tokens]
            next_token = np.argmax(filtered_logits)
            input_ids.append(int(next_token))
            print(f"state={state}, next_token={next_token}, "
                  f"decoded={self.model.decode([next_token])}")

            if state == "FUNCTION_NAME":
                if next_token == self.prefix_structure["FINISHED_FUNCTION"][0]:
                    state = "PARAMS_OPEN"
                else:
                    compatible_function = [function for function
                                           in compatible_function
                                           if function[gen_cont] == next_token]
                    gen_cont += 1
            elif state == "PARAM_VALUE_STRING":
                value_cont += 1
                for k in value_rep:
                    if input_ids[-k:] == input_ids[-2*k:-k]:
                        input_ids = input_ids[:-k]
                        input_ids += self.prefix_structure["FINISHED_FUNCTION"]
                        function = next(f for f in self.functions
                                        if f["name"] == chosen_function)
                        tot_attr = len(function["parameters"])
                        if attr_cont != tot_attr:
                            state = "MULTIPLE_PARAM"
                            value_cont = 0
                        else:
                            state = "END_STRING"
                        break
                else:
                    if self.model.decode([next_token]).endswith('"'):
                        function = next(f for f in self.functions
                                        if f["name"] == chosen_function)
                        tot_attr = len(function["parameters"])
                        if attr_cont != tot_attr:
                            state = "MULTIPLE_PARAM"
                            value_cont = 0
                        else:
                            state = "END_STRING"
            elif state == "PARAM_VALUE_NUMBER":
                value_cont += 1
                if next_token in [self.comma_token]:
                    state = "MULTIPLE_PARAM"
                    value_cont = 0
                elif next_token in [self.close_brace_token]:
                    state = "END_NUMBER"
