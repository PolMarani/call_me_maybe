from llm_sdk import Small_LLM_Model
import argparse
from .file_handler import load_json, write_json
from .constrained_decoder import ConstrainedDecoder


def main() -> None:
    args = parse_arguments()
    function_definition = load_json(args.functions_definition)
    test_prompts = load_json(args.input)

    model = Small_LLM_Model(model_name=args.model)
    constrained_decoder = ConstrainedDecoder(model=model,
                                             functions=function_definition)

    prompt_elab_results = []
    for prompt in test_prompts:
        prompt_text = prompt.get("prompt")
        prompt_elab_results.append(
            constrained_decoder.generate_function_call(prompt_text))

    write_json(prompt_elab_results, args.output)


def parse_arguments() -> argparse.Namespace:
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
        default="data/output/function_calling_result.json",
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
