# Policy Guard - Add policy guardrails to your agent's response
Agent lifecycle components to guard and enforce policies (or instructions) in LLM responses. See as much as 10 point improvement in accuracy (under certain circumstances).

## Table of Contents
- [When to Use This Component](#when-to-use-this-component)
- [Quick Start](#quick-start)
- [License](#license)
- [Under the Hood](#under-the-hood)

## When to Use This Component
There are two primary use cases for policy guards:

1. Reinforce a priori guidelines or policy on LLM output; for example, policy guards can be installed as for specific requirements (e.g., "response should be in table format"), best practices (e.g., "include a personal opening in the email") or constraints ("do not include confidential information")
2. Improve or correct agent responses; for example, if, in spite of including an instruction in the prompt to follow a certain format, say JSON, th agent generates incorrect output, a policy guard can be installed as a corrective agent to enforce a desired behavior.

However, policy guards incur an additional cost and should only be installed when guideline/policy compliance is a priority.


## Quick Start

To illustrate the use of policy guards this folder includes a sample file with LLM responses and policies and a script that shows how to run a policy guard on the LLM responses.

- `examples/sample_responses-llama-policies-10.jsonl`: a file with 100 data samples where each data sample consists of a query and ten policy statements derived from the [IFEval,](https://huggingface.co/datasets/google/IFEval) dataset, as well as LLM responses that were generated with Llama-3.1-70b.
- `examples/run_altk_pipeline.py`: python script to illustrate how to use the policy guard detect and repair components

To run the python script use the following:
```
> python run_altk_pipeline.py --input_file sample_responses-llama-policies-10.jsonl --output_file ouptut.json --verbose
```

The script will produce an output file with detection and repair results.

To see all commandline options use:
```
python run_altk_pipeline.py --help
```

## License
Apache 2.0 - see LICENSE file for details.

## Under the Hood
For more details on how the technology behind this component, the architecture and experimental results, refer to our documentation website (Coming soon).
