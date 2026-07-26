from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model
import numpy as np


class ConstrainedDecoder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Small_LLM_Model
    functions: list
    prefix_structure: dict = {}

    def model_post_init(self, __context: any) -> None:
        """
         prefix_structure e' con [0] perche' con encode ti crea un tensor che e' una lista, e con tolist
         te la mette dentro un'altra lista
        """
        self.prefix_structure = {
            "START": self.model.encode('{').tolist()[0],
            "NAME_DECLARATION": self.model.encode('name": ').tolist()[0],
            "PARAMS_OPEN": self.model.encode('parameters": {').tolist()[0],
            "VALUE_DEFINITION_STRING": self.model.encode('": "').tolist()[0],
            "VALUE_DEFINITION_INT": self.model.encode('":').tolist()[0],
            "MULTIPLE_PARAM": self.model.encode(', "').tolist()[0],
            "END": self.model.encode("}}").tolist()[0]}

    def generate_function_call(self, prompt: str) -> dict:
        input_ids = self.model.encode(prompt).tolist()

        state = "START"
        while True:
            logits = self.model.get_logits_from_input_ids(input_ids)
            if state == "START":
                valid_tokens = self.prefix_structure["START"]
            logits_array = np.array(logits)
            filtered_logits = np.full_like(logits_array, -np.inf)
            filtered_logits[valid_tokens] = logits_array[valid_tokens]
            next_token = np.argmax(filtered_logits)
            input_ids.append(int(next_token))
