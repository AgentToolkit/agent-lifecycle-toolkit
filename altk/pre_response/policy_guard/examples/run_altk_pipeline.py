import argparse
from copy import deepcopy
import json
import os
from dotenv import load_dotenv

from altk.core.llm.base import get_llm
from altk.core.toolkit import ComponentConfig
from altk.pre_response.policy_guard.detect import detector_factory
from altk.pre_response.policy_guard.repair import repairer_factory
from altk.pre_response.policy_guard.detect.task_judge import (
    TaskJudge,
    create_adherence_check_report,
)
from altk.pre_response.policy_guard.core.toolkit import (
    PolicyDetectorInput,
    PolicyDetectorOutput,
    PolicyDetectorSingleOutput,
    PolicyRepairerInput,
    RepairConfig,
)


def get_repair_config(args) -> RepairConfig:
    config = RepairConfig(
        max_retry=args.max_retry,
        max_sample=args.max_sample,
        temperature=args.temperature,
        continue_iterations=args.continue_iterations,
        no_degrade=args.no_degrade,
        allinone=args.allinone,
    )
    return config


def repair_single_datapoint(datapoint: dict, args):
    load_dotenv()
    repair_config = get_repair_config(args)

    WATSONX_CLIENT = get_llm("watsonx")
    llm_client = WATSONX_CLIENT(
        model_id="meta-llama/llama-3-3-70b-instruct",
        api_key=os.getenv("WX_API_KEY"),
        project_id=os.getenv("WX_PROJECT_ID"),
        url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
    )

    config = ComponentConfig(llm_client=llm_client)

    if args.detect_mode == "single":
        detector = detector_factory("single_policy_llm_detector", config)
    elif args.detect_mode == "batch":
        detector = detector_factory("batch_policy_llm_detector", config)

    task_judge = TaskJudge(config=config)
    if args.repair_mode == "batch":
        repairer = repairer_factory("batch_policy_llm_repairer", detector, config)
    elif args.repair_mode == "retry":
        repairer = repairer_factory("retry_llm_repairer", detector, config)
    elif args.repair_mode == "bestofn":
        repairer = repairer_factory("bestofn_llm_repairer", detector, config)
    elif args.repair_mode == "bestofngen":
        repairer = repairer_factory("bestofn_llm_generator", detector, config)
    elif args.repair_mode == "mapreduce":
        repairer = repairer_factory("mapreduce_llm_repairer", detector, config)
    else:
        repairer = repairer_factory("iterative_llm_repairer", detector, config)

    # Create the output dictionary
    single_result = {}
    KEYS_TO_COPY = ["key", "prompt_request", "instruction_text_list"]
    for key in KEYS_TO_COPY:
        single_result[key] = deepcopy(datapoint[key])

    single_result["failure"] = None
    try:
        response = datapoint["response"]
        if args.verbose:
            print("\n++++++ READ RESPONSE:\n", response + "\n\n")

        single_result["original_response"] = response
        # Check task adherence.
        if not args.no_adhere_check:
            # first check whether we can read in the adherence check from the input file
            if "pre_task_adherence" in datapoint:
                single_result["pre_task_adherence"] = datapoint["pre_task_adherence"]
                if args.verbose:
                    print("\n++++++ READING TASK ADHERENCE CHECK:")
            else:
                single_result["pre_task_adherence"] = task_judge.check_task_completion(
                    single_result["prompt_request"], response
                )
                if args.verbose:
                    print("\n++++++ TASK ADHERENCE CHECK:")
            if args.verbose:
                print(json.dumps(single_result["pre_task_adherence"], indent=4))

            if single_result["pre_task_adherence"]["score"] != "Yes":
                # skip non-passing responses
                if args.verbose:
                    print(
                        "\n++++++ TASK ADHERENCE CHECK: skipping non-passing response"
                    )
                single_result["failure"] = "FAILED TASK ADHERENCE CHECK"
                return single_result

        # Get ground truth violations
        input_example = PolicyDetectorInput(
            policies=single_result["instruction_text_list"],
            prompt=single_result["prompt_request"],
            response=response,
        )

        # Get violations from implemented detector
        if not args.allinone:
            detection = detector.detect(input_example)
            detected_compliance = [d.compliance for d in detection.policy_outputs]

            if args.verbose:
                print("\n" + "++++++ DETECTED COMPLIANCE: ", detected_compliance)
            #    print('\n' + "++++++ GROUND TRUTH COMPLIANCE: ", gt_compliance)
            single_result["initial_detection"] = [
                d.model_dump() for d in detection.policy_outputs
            ]
        else:
            detection = PolicyDetectorOutput(
                policy_outputs=[
                    PolicyDetectorSingleOutput(
                        policy=p, compliance=False, explanation=""
                    )
                    for p in single_result["instruction_text_list"]
                ]
            )
            detected_compliance = [
                False for _ in range(len(single_result["instruction_text_list"]))
            ]

        repair_response = response
        if args.allinone or (
            False in detected_compliance
            and args.repair_mode not in ["None", "none", None]
        ):
            repair_inputs = PolicyRepairerInput(
                config=repair_config,
                detection_input=input_example,
                detection_output=detection,
            )

            repair = repairer.repair(repair_inputs)
            repair_response = repair.repaired_text

            input_example.response = repair_response

            if args.verbose:
                print("\n++++++ REPAIRED RESPONSE:\n", repair_response)

            single_result["repair_response"] = repair_response
            single_result["bestofn_attempts"] = repair.bestofn_attempts
            single_result["repair_mode"] = args.repair_mode

            # Check task adherence.
            if not args.no_adhere_check:
                single_result["post_task_adherence"] = task_judge.check_task_completion(
                    single_result["prompt_request"], repair_response
                )
                if args.verbose:
                    print("\n++++++ REPAIRED TASK ADHERENCE CHECK:")
                    print(json.dumps(single_result["post_task_adherence"], indent=4))
        else:
            single_result["repair_response"] = repair_response
            if not args.no_adhere_check:
                single_result["post_task_adherence"] = single_result[
                    "pre_task_adherence"
                ]

    except Exception as e:
        print(f"\n++++++ EXCEPTION ERROR: {e}")
        import traceback

        traceback.print_exc()
        single_result["failure"] = str(e)

    return single_result


def main(args):
    # Double check some stuff before running anything
    assert os.path.isfile(args.input_file)

    data = []
    infile = args.input_file
    with open(infile, "r") as f:
        if infile.endswith(".json"):
            data = json.load(f)
        else:
            for line in f:
                dat = json.loads(line)
                data.append(dat)

    results = []

    if args.num_datapoints > 0:
        data = data[: args.num_datapoints]

    for d in data:
        try:
            result = repair_single_datapoint(d, args)
        except Exception as e:
            result = {
                "input": d,
                "valid": False,
                "failure": f"Unhandled exception: {str(e)}",
            }
            print(f"ERROR: {result['failure']}")
        results.append(result)
        print(f"Finished {len(results)} / {len(data)} queries. ")

    # Separate out a list of runtime errors
    failures = [
        {"index": i, "failure": s["failure"]}
        for i, s in enumerate(results)
        if s["failure"] is not None
    ]
    successes = [s for s in results if s["failure"] is None]

    if not args.no_adhere_check:
        task_adherence_report = create_adherence_check_report(successes)
    else:
        task_adherence_report = {}

    print("\n+++++++++++++++++++++++++")
    print("FAILURES: ", len(failures))
    print("PROCESSED QUERIES: ", len(results))
    print("TASK ADHERENCE RESULTS:")
    print(json.dumps(task_adherence_report, indent=4))

    full_output = {
        "data": results,
        "failures": len(failures),
        "error_report": failures,
        "task_adherence_summary": task_adherence_report,
        "runtime_args": vars(args),
    }
    with open(args.output_file, "w") as f:
        json.dump(full_output, f)


def create_parser():
    parser = argparse.ArgumentParser(description="Output/repair")
    parser.add_argument(
        "-if", "--input_file", type=str, help="topic file", required=True
    )
    parser.add_argument(
        "-of", "--output_file", type=str, help="output file", default="output.json"
    )
    parser.add_argument(
        "-rm",
        "--repair_mode",
        type=str,
        help="repair mode",
        default="batch",
        choices=[
            "batch",
            "iterative",
            "retry",
            "none",
            "bestofn",
            "bestofngen",
            "mapreduce",
        ],
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="temperature used with best-of-n repairer, or generator when repeat > 1",
        default=0.5,
    )
    parser.add_argument(
        "-n",
        "--num_datapoints",
        type=int,
        help="max datapoints, -1 to run all",
        default=-1,
    )
    parser.add_argument(
        "-mr", "--max_retry", type=int, help="max number of retry iterations", default=5
    )
    parser.add_argument(
        "-ms",
        "--max_sample",
        type=int,
        help="max number of sampled responses generated for best-of-n",
        default=5,
    )
    parser.add_argument(
        "--continue_iterations",
        action="store_true",
        help="For best-of-n, should it continue even if it finds a solution.",
    )
    parser.add_argument(
        "-we",
        "--with_explanations",
        action="store_true",
        help="use explanations in repair promtps",
    )
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose")
    parser.add_argument("--allinone", action="store_true", dest="allinone")
    parser.add_argument(
        "--no_degrade",
        action="store_true",
        help="roll back repair attempts that include a self-detected degradion",
    )
    parser.add_argument(
        "--no_adhere_check",
        action="store_true",
        help="don't include task adherence checks",
    )
    parser.add_argument(
        "-dm",
        "--detect_mode",
        type=str,
        help="detection mode",
        default="batch",
        choices=["batch", "single"],
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if args.verbose:
        print(f"Arguments: {args}\n")
    main(args)
