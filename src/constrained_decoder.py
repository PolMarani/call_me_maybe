from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model
import numpy as np


class ConstrainedDecoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Small_LLM_Model
    functions: list
    prefix_structure: dict = {}
    function_name_tokens: list = []
    attributes_names_tokens: dict = {}

    def model_post_init(self, __context: any) -> None:
        self.prefix_structure = {
            "START": self.model.encode('{').tolist()[0],
            "NAME_KEY": self.model.encode('"name": "').tolist()[0],
            "FINISHED_FUNCTION": self.model.encode('"').tolist()[0],
            "PARAMS_OPEN": self.model.encode('parameters": {').tolist()[0],
            "VALUE_DEFINITION_STRING": self.model.encode('": "').tolist()[0],
            "VALUE_DEFINITION_INT": self.model.encode('":').tolist()[0],
            "MULTIPLE_PARAM": self.model.encode(', "').tolist()[0],
            "END": self.model.encode("}}").tolist()[0]}
        self.function_name_tokens = [
            self.model.encode(function["name"].tolist())
            for function in self.functions]
        # Crea dict di liste, che sarebbero gli attributi per ogni funzione
        self.attributes_names_tokens = {
            function["name"]:
            [self.model.encode(param_name).tolist()[0]
             for param_name in function["parameters"].keys()]
            for function in self.functions
        }

    def generate_function_call(self, prompt: str) -> dict:
        input_ids = self.model.encode(prompt).tolist()[0]
        compatible_function = self.function_name_tokens.copy()
        gen_cont = 0
        attr_cont = 0
        state = "START"
        chosen_function = None

        while True:
            logits = self.model.get_logits_from_input_ids(input_ids)

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
                if (gen_cont == len(compatible_function[0])):
                    valid_tokens = self.prefix_structure["FINISHED_FUNCTION"]
                    chosen_function = self.model.decode(compatible_function[0])
                else:
                    valid_tokens = [function[gen_cont]
                                    for function in compatible_function]
            # state FUNCTION_NAME
            elif state == "PARAMS_OPEN":
                input_ids += self.prefix_structure["PARAMS_OPEN"]
                state = "PARAM_NAME"
                continue
            elif state == "PARAM_NAME":
                function = next(f for f in self.functions if f["name"] == chosen_function)
                attribute = list(function["parameters"].keys())[attr_cont]
                input_ids += self.attributes_names_tokens[chosen_function][attr_cont]
                attr_type = function["parameters"][attribute]["type"]
                attr_cont += 1
                if len()
                    

            #  if (len(self.attributes_names_tokens[chosen_function]) != attr_cont):
            
            logits_array = np.array(logits)
            filtered_logits = np.full_like(logits_array, -np.inf)
            filtered_logits[valid_tokens] = logits_array[valid_tokens]
            next_token = np.argmax(filtered_logits)
            input_ids.append(int(next_token))

            if state == "FUNCTION_NAME":
                if next_token == self.prefix_structure["FINISHED_FUNCTION"][0]:
                    state = "PARAMS_OPEN"
                else:
                    compatible_function = [function for function
                                           in compatible_function
                                           if function[gen_cont] == next_token]
                    gen_cont += 1
