from pydantic import BaseModel, ConfigDict
from llm_sdk import Small_LLM_Model
from typing import Any
import numpy as np
import textwrap
import json
import re

GREEN = "\033[32m"
BLUE = "\033[94m"
RESET = "\033[0m"


class ConstrainedDecoder(BaseModel):
    """Translate natural language prompts into structured function calls.

    Uses constrained decoding to guarantee that the underlying LLM's
    output is always valid, schema-compliant JSON: a finite state
    machine tracks which part of the JSON structure is being
    generated, and at every model-driven step the logits for tokens
    outside the currently valid set are masked to `-inf` before
    picking the next token.

    Attributes
    ----------
    model : Small_LLM_Model
        The wrapped language model used to encode text and generate
        logits.
    functions : list
        The list of available function definitions (name, parameters,
        types) that the model can choose to call.
    prefix_structure : dict
        Pre-tokenized fixed fragments of the JSON output (e.g. the
        opening brace, the `"name": "` key, parameter separators),
        computed once in `model_post_init`.
    function_name_tokens : list
        Pre-tokenized name of every available function, used for
        prefix matching during function-name generation.
    attributes_names_tokens : dict
        Pre-tokenized parameter names for every function, keyed by
        function name.
    number_tokens : list
        Token IDs whose text is compatible with a JSON number
        (digits, `.`, `-`, `e`, `E`), plus the comma and closing
        brace tokens.
    string_tokens : list
        Token IDs whose text does not contain a `"` character, plus
        the closing-quote token.
    comma_token : int
        Token ID for the `,` character.
    close_brace_token : int
        Token ID for the `}` character.
    """

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
        """Pre-compute all fixed tokens and vocabulary-derived data.

        Runs once, right after the Pydantic model is initialized.
        Pre-tokenizes every fixed JSON fragment, every function name,
        and every parameter name, and builds the sets of token IDs
        that are valid inside a JSON string or number, by reading the
        model's own vocabulary file. Doing this once here (instead of
        on every call to `generate_function_call`) keeps per-prompt
        generation fast.

        Parameters
        ----------
        __context : Any
            Pydantic's post-init context object. Unused, but required
            by the `model_post_init` signature.

        Returns
        -------
        None
        """
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
        """Clean up common malformed patterns in a generated regex string.

        Small models occasionally produce a `regex` argument that is
        syntactically valid but semantically wrong. This applies two
        narrow, targeted corrections for the failure modes observed
        in practice, without touching patterns that do not match
        those exact shapes:

        1. A dangling trailing `|` (empty alternation), which in
           Python's `re` module matches at every position and breaks
           the substitution — it is stripped.
        2. A trailing `.*` directly appended to a single plain word
           (e.g. `cat.*`), which greedily matches everything after
           the word instead of just the word itself — the `.*` is
           stripped, leaving only the word.

        Parameters
        ----------
        par_value : str
            The raw regex pattern generated by the model.

        Returns
        -------
        str
            The corrected pattern, or the original pattern unchanged
            if neither correction applies.
        """
        stripped = par_value.rstrip()
        if stripped.endswith('|'):
            stripped = stripped.rstrip('|')
        if stripped.endswith('.*'):
            m = re.fullmatch(r'([A-Za-z0-9_]+)\.\*', stripped)
            if m:
                stripped = m.group(1)
        return stripped

    def fix_repeated_replacement(self, value: str) -> str:
        """Collapse a replacement value made of one character repeated.

        Small models sometimes over-generate a `replacement` argument as
        the same character repeated several times (e.g. `"*****"`) when
        a single occurrence was intended (e.g. for "replace X with an
        asterisk"). If `value` consists of exactly one distinct
        character repeated more than once, only the first character is
        kept; any other string (including a single character, an empty
        string, or a string with more than one distinct character) is
        returned unchanged.

        Parameters
        ----------
        value : str
            The raw `replacement` value generated by the model.

        Returns
        -------
        str
            A single instance of the repeated character, or the
            original `value` unchanged if it isn't made of one character
            repeated.
        """
        if len(set(value)) == 1 and len(value) > 1:
            return value[0]
        else:
            return value

    def generate_function_call(self, prompt: str) -> dict:
        """Generate a structured function call for a natural language prompt.

        Runs the constrained-decoding finite state machine end to
        end: builds the model's context (instructions, available
        functions, and the user's prompt), then generates the
        function-call JSON token by token, restricting the model to
        only the tokens that keep the output valid at each step
        (fixed JSON structure, a real function name, and correctly
        typed argument values). Guards against greedy-decoding
        repetition loops and enforces hard length caps on generated
        values as a safety net.

        Parameters
        ----------
        prompt : str
            The natural language request to translate into a
            function call (e.g. "What is the sum of 2 and 3?").

        Returns
        -------
        dict
            A dictionary with exactly three keys: `prompt` (the
            original input), `name` (the chosen function's name),
            and `parameters` (a dict of argument name/value pairs
            with the correct types).
        """
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
        result: dict[str, Any] = {}
        attr_type = ""
        value_cont = 0
        value_rep = [2, 3, 4, 5, 6, 7]

        print(f"{BLUE} - The prompt is:{RESET}", prompt)
        print()
        print(f"{BLUE}Searching for the matching function...{RESET}")
        while True:

            if state == "START":
                input_ids += self.prefix_structure["START"]
                state = "NAME_KEY"
                continue
            elif state == "NAME_KEY":
                input_ids += self.prefix_structure["NAME_KEY"]
                state = "FUNCTION_NAME"
                continue
            elif state == "FUNCTION_NAME":
                logits = self.model.get_logits_from_input_ids(input_ids)
                if (gen_cont == len(compatible_function[0])):
                    valid_tokens = self.prefix_structure["FINISHED_FUNCTION"]
                    chosen_function = self.model.decode(compatible_function[0])
                else:
                    valid_tokens = [function[gen_cont]
                                    for function in compatible_function]
            elif state == "PARAMS_OPEN":
                input_ids += self.prefix_structure["PARAMS_OPEN"]
                state = "PARAM_NAME"
                continue
            elif state == "PARAM_NAME":
                logits = self.model.get_logits_from_input_ids(input_ids)
                function = next(f for f in self.functions
                                if f["name"] == chosen_function)
                if attr_cont == 0:
                    print(f"function found: \n {function}")
                    print(f"parameters: \n {function['parameters']}")
                    print(f"{BLUE}Generating argument values...{RESET}")
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
                if "replacement" in result["parameters"]:
                    result["parameters"]["replacement"] = (
                        self.fix_repeated_replacement(
                            result["parameters"]["replacement"]))
                print(f"{GREEN} ->{RESET}")
                print(f"{GREEN}RESULT IS:{RESET}", result)
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
