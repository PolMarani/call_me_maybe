from llm_sdk import Small_LLM_Model
import argparse
from .file_handler import load_json, write_json
from .constrained_decoder import ConstrainedDecoder
import sys


def main() -> None:
    """Run the function-calling pipeline end to end.

    Loads the function definitions and test prompts from the paths
    given on the command line (or their defaults), loads the chosen
    LLM, generates a structured function call for every prompt using
    constrained decoding, and writes all results to the output file.

    Returns
    -------
    None
        This function does not return a value; results are written
        to disk via `write_json`.
    """
    args = parse_arguments()
    function_definition = load_json(args.functions_definition)
    if not function_definition:
        print("ERROOOOOOOOOORRR on function definition")
        sys.exit(1)
    test_prompts = load_json(args.input)
    if not test_prompts:
        print("ERROOOOOOOOOORRR on test prompt file")
        sys.exit(1)
    try:
        model = Small_LLM_Model(model_name=args.model)
    except Exception as e:
        print(f"ERROOOOOOOOOORRR loading model '{args.model}': {e}")
        sys.exit(1)
    constrained_decoder = ConstrainedDecoder(model=model,
                                             functions=function_definition)

    prompt_elab_results = []
    for prompt in test_prompts:
        prompt_text = prompt.get("prompt")
        prompt_elab_results.append(
            constrained_decoder.generate_function_call(prompt_text))

    write_json(prompt_elab_results, args.output)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the function-calling program.

    Returns
    -------
    argparse.Namespace
        The parsed arguments, exposing `functions_definition`,
        `input`, `output`, and `model` as attributes.
    """
    parser = argparse.ArgumentParser(
        description="Function calling with LLM using constrained decoding"
    )
    parser.add_argument(
        '--functions_definition', type=str,
        default="data/input/functions_definition.json",
        help="Path to JSON file containing function definitions"
    )
    parser.add_argument(
        '--input', type=str,
        default="data/input/function_calling_tests.json",
        help="Path to JSON file containing test prompts"
    )
    parser.add_argument(
        '--output', type=str,
        default="data/output/function_calling_results.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        '--model', type=str,
        default="Qwen/Qwen3-0.6B",
        help="Choose the model to use"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
