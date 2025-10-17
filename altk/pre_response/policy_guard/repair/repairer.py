from abc import abstractmethod
import copy
from typing import Set

from altk.core.toolkit import ComponentBase, ComponentConfig, AgentPhase
from altk.pre_response.policy_guard.core.utils import generate_params
from altk.pre_response.policy_guard.detect.detector import Detector
from altk.pre_response.policy_guard.repair.repair_prompts import (
    priority_repair_prompt,
    ordered_repair_prompt,
    explanation_repair_prompt,
    query_repair_prompt,
    query_gen_prompt,
    query_single_repair_prompt,
    allinone_repair_prompt,
    allinone_single_repair_prompt,
    mapreduce_repair_prompt,
)
from altk.pre_response.policy_guard.core.toolkit import (
    PolicyRepairerInput,
    PolicyRepairerOutput,
    PolicyDetectorInput,
)


BATCH_REPAIR_NAME = "batch_policy_llm_repairer"
ITERATIVE_REPAIR_NAME = "iterative_llm_repairer"
RETRY_REPAIR_NAME = "retry_llm_repairer"
BESTOFN_REPAIR_NAME = "bestofn_llm_repairer"
BESTOFNGEN_REPAIR_NAME = "bestofn_llm_generator"
MAPREDUCE_REPAIR_NAME = "mapreduce_llm_repairer"


class Repairer(ComponentBase):
    # LLM config here
    config: ComponentConfig
    name: str
    detector: Detector

    def repair_endpoint(self, prompt: str) -> str:
        response = self.config.llm_client.generate(prompt)
        return response

    @classmethod
    def supported_phases(cls) -> Set[AgentPhase]:
        """
        The repairer middleware is only supported at runtime
        Returns:
            Set[AgentPhase]: A set of supported AgentPhase.
        """
        return {AgentPhase.RUNTIME}

    def process(
        self, data: PolicyRepairerInput, phase: AgentPhase
    ) -> PolicyRepairerOutput:
        """
        Detect policy violations.
        Args:
            data (PolicyRepairerInput): Input data for the middleware.
            phase (AgentPhase): The phase in which the middleware is being executed.

        Returns:
            PolicyRepairerOutput: Processed output data.
        Raises:
            ValueError: If the phase is not supported by the middleware.
            NotImplementedError: If the phase is not handled in the middleware.
        """
        if phase not in self.supported_phases():
            raise ValueError(
                f"{self.__class__.__name__} does not support phase {phase}"
            )

        if phase == AgentPhase.BUILDTIME:
            return self._build(data)
        elif phase == AgentPhase.RUNTIME:
            return self._run(data)

        raise NotImplementedError(f"Unhandled phase: {phase}")

    def _run(self, data: PolicyRepairerInput) -> PolicyRepairerOutput:
        """
        This method is called during the RUNTIME phase.
        Args:
            data (PolicyRepairerInput): Input data for the run phase.
        """
        return self.repair(data)

    @abstractmethod
    def repair(self, repair_input: PolicyRepairerInput) -> PolicyRepairerOutput:
        raise NotImplementedError("Subclass must implement `repair` method. ")

    def postprocess_repair_text(self, text: str) -> str:
        postprocessed_text = (
            text.replace("<START_OF_REWRITE>", "")
            .replace("<END_OF_REWRITE>", "")
            .strip()
        )
        return postprocessed_text

    def postprocess_gen_text(self, text: str) -> str:
        postprocessed_text = (
            text.replace("<START_OF_RESPONSE>", "")
            .replace("<END_OF_RESPONSE>", "")
            .strip()
        )
        return postprocessed_text

    def select_bestofn(self, results: list) -> list:
        best_j = 0
        # results[0] is the original to-be-repaired response
        for j in range(len(results)):
            if results[j]["compliances"].count(True) > results[best_j][
                "compliances"
            ].count(True):
                best_j = j

        # print(f"\nSelect best-of-n: selecting best version {best_j} with compliances {results[best_j]['compliances']}")
        return results[best_j]["response"]

    def select_bestofn_nodegrade(self, results: list) -> list:
        best_j = 0
        # we are measuring degradations against results[0], the original to-be-repaired response
        for j in range(len(results)):
            if results[j]["compliances"].count(True) > results[best_j][
                "compliances"
            ].count(True):
                degradation_list = [
                    (c_initial and not c_repair)
                    for c_initial, c_repair in zip(
                        results[0]["compliances"], results[j]["compliances"]
                    )
                ]
                if not any(degradation_list):
                    best_j = j
        # print(f"Select best-of-n: selecting best non-degrading version {best_j} with compliances {results[best_j]['compliances']}")
        return results[best_j]["response"]


class BatchPolicyRepairer(Repairer):
    name: str = BATCH_REPAIR_NAME

    def repair(self, repair_input: PolicyRepairerInput) -> PolicyRepairerOutput:
        assert not (
            (repair_input.weights is not None) and (repair_input.ranks is not None)
        ), "Cannot order the repair with both weights and ranks. "
        all_policies = [r.policy for r in repair_input.detection_output.policy_outputs]
        all_compliances = [
            r.compliance for r in repair_input.detection_output.policy_outputs
        ]
        all_explanations = [
            r.explanation for r in repair_input.detection_output.policy_outputs
        ]
        text = repair_input.detection_input.response

        results = []
        # collect initial response for possible roll-back
        result = {"response": text, "compliances": all_compliances}
        results.append(result)

        # Filter policies that don't need fixing.
        policies = [p for p, c in zip(all_policies, all_compliances) if not c]
        explanations = [e for e, c in zip(all_explanations, all_compliances) if not c]
        followed_policies = [p for p, c in zip(all_policies, all_compliances) if c]

        if repair_input.weights is not None:
            weights = [
                w for w, c in zip(repair_input.weights, all_compliances) if not c
            ]
            prompt = ordered_repair_prompt(text, policies, weights=weights)
        elif repair_input.ranks is not None:
            ranks = [r for r, c in zip(repair_input.ranks, all_compliances) if not c]
            prompt = ordered_repair_prompt(text, policies, ranks=ranks)
        elif repair_input.tags is not None:
            tags = [t for t, c in zip(repair_input.tags, all_compliances) if not c]
            prompt = priority_repair_prompt(text, policies, tags)
        elif len(explanations) > 0 and not all([e == "" for e in explanations]):
            prompt = explanation_repair_prompt(
                self.config.llm_client.model_id,
                text,
                followed_policies,
                policies,
                repair_input.detection_input.prompt,
                explanations,
            )
        elif repair_input.config.allinone:
            prompt = allinone_repair_prompt(
                self.config.llm_client.model_id,
                text,
                all_policies,
                repair_input.detection_input.prompt,
            )
        else:
            #   prompt = query_repair_prompt(self.config.llm_client.model_id, text, inputs.instruction_text_list, policies, repair_input.detection_input.prompt)
            prompt = query_repair_prompt(
                self.config.llm_client.model_id,
                text,
                followed_policies,
                policies,
                repair_input.detection_input.prompt,
            )

        repaired_text = self.repair_endpoint(prompt)
        repaired_text = self.postprocess_repair_text(repaired_text)

        if repair_input.config.no_degrade:
            # Shortlist detector inputs
            detector_inputs = PolicyDetectorInput(
                policies=all_policies,
                prompt=repair_input.detection_input.prompt,
                response=repaired_text,
            )

            detection = self.detector.detect(detector_inputs)
            compliance = [d.compliance for d in detection.policy_outputs]

            result = {
                "response": repaired_text,
                "compliances": compliance,
            }
            results.append(result)
            repaired_text = self.select_bestofn_nodegrade(results)

        return PolicyRepairerOutput(
            repaired_text=repaired_text, bestofn_attempts=results
        )


class IterativeRepairer(Repairer):
    name: str = ITERATIVE_REPAIR_NAME

    def repair(self, repair_input: PolicyRepairerInput) -> PolicyRepairerOutput:
        assert not (
            (repair_input.weights is not None) and (repair_input.ranks is not None)
        ), "Cannot order the repair with both weights and ranks. "
        text = repair_input.detection_input.response
        policies = repair_input.detection_input.policies

        if repair_input.weights is not None:
            combined = list(
                zip(
                    repair_input.weights,
                    policies,
                    [
                        r.compliance
                        for r in repair_input.detection_output.policy_outputs
                    ],
                )
            )
            combined.sort(key=lambda x: x[0], reverse=True)
            weights_sorted, policies_sorted, compliance_sorted = zip(*combined)
        elif repair_input.ranks is not None:
            combined = list(
                zip(
                    repair_input.ranks,
                    policies,
                    [
                        r.compliance
                        for r in repair_input.detection_output.policy_outputs
                    ],
                )
            )
            combined.sort(key=lambda x: x[0], reverse=True)
            ranks_sorted, policies_sorted, compliance_sorted = zip(*combined)
        else:
            policies_sorted = policies
            compliance_sorted = [
                r.compliance for r in repair_input.detection_output.policy_outputs
            ]

        compliances = compliance_sorted
        policies = policies_sorted
        repaired_text = text
        results = []
        if repair_input.config.no_degrade:
            # initialize with to be repaired text for possible roll-back
            result = {
                "response": repaired_text,
                "compliances": compliances,
            }
            results.append(result)

        for idx in range(len(policies)):
            nodetect = all(
                [
                    c.compliance == ""
                    for c in repair_input.detection_output.policy_outputs
                ]
            )
            if not nodetect:  # if don't have detections, we can't skip compliant ones
                if compliances[idx]:
                    # If it ain't broke, don't fix it.
                    continue

            policy = policies[idx]
            followed_policies = [
                p for p, c in zip(policies[:idx], compliances[:idx]) if c
            ]

            if repair_input.config.allinone:
                prompt = allinone_single_repair_prompt(
                    self.config.llm_client.model_id,
                    repaired_text,
                    policies,
                    policy,
                    repair_input.detection_input.prompt,
                )
            # TODO: sort these above to enable
            # elif repair_input.explanations is not None and len(repair_input.explanations) > 0:
            #     explanation = repair_input.explanations[idx]
            #     prompt = explanation_single_repair_prompt(self.config.llm_client.model_id, repaired_text, followed_policies, policy, repair_input.detection_input.prompt, explanation)
            else:
                prompt = query_single_repair_prompt(
                    self.config.llm_client.model_id,
                    repaired_text,
                    followed_policies,
                    policy,
                    repair_input.detection_input.prompt,
                )

            candidate_repaired_text = self.repair_endpoint(
                prompt, self.config.llm_client.model_id, return_counts=False
            )
            candidate_repaired_text = self.postprocess_repair_text(
                candidate_repaired_text
            )

            detector_inputs = PolicyDetectorInput(
                policies=policies,
                prompt=repair_input.detection_input.prompt,
                response=candidate_repaired_text,
            )

            detections = self.detector.detect(detector_inputs)
            compliances = [d.compliance for d in detections.policy_outputs]

            # collect results
            result = {
                "response": candidate_repaired_text,
                "compliances": compliances,
            }
            results.append(result)

            repaired_text = candidate_repaired_text

        # pick the best repaired response
        if repair_input.config.no_degrade:
            repaired_text = self.select_bestofn_nodegrade(results)
        else:
            repaired_text = self.select_bestofn(results)

        return PolicyRepairerOutput(
            repaired_text=repaired_text, bestofn_attempts=results
        )


class RetryRepairer(Repairer):
    name: str = RETRY_REPAIR_NAME

    def repair(self, repair_input: PolicyRepairerInput) -> PolicyRepairerOutput:
        all_policies = repair_input.detection_input.policies
        compliance = [
            d.compliance for d in repair_input.detection_output.policy_outputs
        ]
        repaired_text = repair_input.detection_input.response

        # Filter policies that don't need fixing.
        retry_policies = [p for p, c in zip(all_policies, compliance) if not c]
        explanations = [
            e
            for e, c in zip(
                [d.explanation for d in repair_input.detection_output.policy_outputs],
                compliance,
            )
            if not c
        ]
        followed_policies = [p for p, c in zip(all_policies, compliance) if c]

        retry_cnt = 0
        results = []
        retry_compliance = compliance

        if repair_input.config.no_degrade:
            # initialize with to be repaired text for possible roll-back
            retry_result = {
                "response": repaired_text,
                "compliances": retry_compliance,
            }
            results.append(retry_result)

        while retry_cnt < repair_input.config.max_retry:
            retry_cnt += 1

            if repair_input.config.allinone:
                prompt = allinone_repair_prompt(
                    self.config.llm_client.model_id,
                    repaired_text,
                    all_policies,
                    repair_input.detection_input.prompt,
                )
            elif len(explanations) > 0 and not all([e == "" for e in explanations]):
                prompt = explanation_repair_prompt(
                    self.config.llm_client.model_id,
                    repaired_text,
                    followed_policies,
                    retry_policies,
                    repair_input.detection_input.prompt,
                    explanations,
                )
            else:
                prompt = query_repair_prompt(
                    self.config.llm_client.model_id,
                    repaired_text,
                    followed_policies,
                    retry_policies,
                    repair_input.detection_input.prompt,
                )

            repaired_text = self.repair_endpoint(
                prompt, self.config.llm_client.model_id, return_counts=False
            )
            repaired_text = self.postprocess_repair_text(repaired_text)

            detector_inputs = PolicyDetectorInput(
                policies=all_policies,
                prompt=repair_input.detection_input.prompt,
                response=repaired_text,
            )

            retry_detection = self.detector.detect(detector_inputs)
            retry_compliance = [d.compliance for d in retry_detection.policy_outputs]
            retry_explanations = [d.explanation for d in retry_detection.policy_outputs]

            # Filter policies that don't need fixing.
            retry_policies = [
                p for p, c in zip(all_policies, retry_compliance) if not c
            ]
            explanations = [
                e for e, c in zip(retry_explanations, retry_compliance) if not c
            ]
            followed_policies = [p for p, c in zip(all_policies, retry_compliance) if c]

            # collect retry results
            retry_result = {
                "response": repaired_text,
                "compliances": retry_compliance,
            }
            results.append(retry_result)

        if retry_cnt > 0:
            # pick the best repaired resppnse
            if repair_input.config.no_degrade:
                repaired_text = self.select_bestofn_nodegrade(results)
            else:
                repaired_text = self.select_bestofn(results)

        return PolicyRepairerOutput(
            repaired_text=repaired_text, bestofn_attempts=results
        )


class BestofNRepairer(Repairer):
    name: str = BESTOFN_REPAIR_NAME

    def repair(self, repair_input: PolicyRepairerInput) -> PolicyRepairerOutput:
        all_policies = repair_input.detection_input.policies
        compliance = [
            d.compliance for d in repair_input.detection_output.policy_outputs
        ]
        explanations = [
            d.explanation for d in repair_input.detection_output.policy_outputs
        ]
        text = repair_input.detection_input.response

        results = []
        if repair_input.config.no_degrade:
            # initialize with to be repaired text for possible roll-back
            result = {
                "response": text,
                "compliances": compliance,
            }
            results.append(result)

        if repair_input.config.allinone:
            prompt = allinone_repair_prompt(
                self.config.llm_client.model_id,
                text,
                all_policies,
                repair_input.detection_input.prompt,
            )
        else:
            # Filter policies that don't need fixing.
            policies = [p for p, c in zip(all_policies, compliance) if not c]
            explanations = [e for e, c in zip(explanations, compliance) if not c]
            followed_policies = [p for p, c in zip(all_policies, compliance) if c]
            if len(explanations) > 0 and not all([e == "" for e in explanations]):
                prompt = explanation_repair_prompt(
                    self.config.llm_client.model_id,
                    text,
                    followed_policies,
                    policies,
                    repair_input.detection_input.prompt,
                    explanations,
                )
            else:
                prompt = query_repair_prompt(
                    self.config.llm_client.model_id,
                    text,
                    followed_policies,
                    policies,
                    repair_input.detection_input.prompt,
                )

        # produce a base sample with default paramaters
        repaired_text = self.repair_endpoint(
            prompt, self.config.llm_client.model_id, return_counts=False
        )
        repaired_text = self.postprocess_repair_text(repaired_text)

        detector_inputs = PolicyDetectorInput(
            policies=all_policies,
            prompt=repair_input.detection_input.prompt,
            response=repaired_text,
        )

        detection = self.detector.detect(detector_inputs)
        compliance = [d.compliance for d in detection.policy_outputs]

        # collect retry results
        result = {
            "response": repaired_text,
            "compliances": compliance,
        }
        results.append(result)

        params = generate_params(
            decoding_method="sample", temperature=repair_input.config.temperature
        )
        # now sample more repair responses at higher temperatures
        for _i in range(repair_input.config.max_sample - 1):
            repaired_text = self.repair_endpoint(
                prompt,
                self.config.llm_client.model_id,
                params=params,
                return_counts=False,
            )
            repaired_text = self.postprocess_repair_text(repaired_text)

            detector_inputs = PolicyDetectorInput(
                policies=all_policies,
                prompt=repair_input.detection_input.prompt,
                response=repaired_text,
            )

            detection = self.detector.detect(detector_inputs)
            compliance = [d.compliance for d in detection.policy_outputs]

            if (
                compliance.count(False) == 0
                and not repair_input.config.continue_iterations
            ):
                # successful repair, no need to sample further
                result = {"response": repaired_text, "compliances": compliance}
                results.append(result)
                break

            # collect retry results
            result = {
                "response": repaired_text,
                "compliances": compliance,
            }
            results.append(result)

        # pick the best repaired response
        if repair_input.config.no_degrade:
            repaired_text = self.select_bestofn_nodegrade(results)
        else:
            repaired_text = self.select_bestofn(results)

        return PolicyRepairerOutput(
            repaired_text=repaired_text, bestofn_attempts=results
        )


class BestofNGenerator(Repairer):
    name: str = BESTOFNGEN_REPAIR_NAME

    def repair(self, repair_input: PolicyRepairerInput) -> PolicyRepairerOutput:
        policies = repair_input.detection_input.policies
        text = repair_input.detection_input.response

        results = []
        prompt = query_gen_prompt(
            self.config.llm_client.model_id,
            policies,
            repair_input.detection_input.prompt,
        )

        detector_inputs = PolicyDetectorInput(
            policies=policies, prompt=repair_input.detection_input.prompt, response=text
        )

        detection = self.detector.detect(detector_inputs)
        compliance = [d.compliance for d in detection.policy_outputs]

        # collect results
        result = {
            "response": text,
            "compliances": compliance,
        }
        results.append(result)

        params = generate_params(
            decoding_method="sample", temperature=repair_input.config.temperature
        )
        # now sample more repair responses at higher temperatures
        for _i in range(repair_input.config.max_sample - 1):
            repaired_text = self.repair_endpoint(
                prompt,
                self.config.llm_client.model_id,
                params=params,
                return_counts=False,
            )
            repaired_text = self.postprocess_gen_text(repaired_text)

            detector_inputs = PolicyDetectorInput(
                policies=policies,
                prompt=repair_input.detection_input.prompt,
                response=repaired_text,
            )

            detection = self.detector.detect(detector_inputs)
            compliance = [d.compliance for d in detection.policy_outputs]

            if (
                compliance.count(False) == 0
                and not repair_input.config.continue_iterations
            ):
                # successful repair, no need to sample further
                result = {"response": repaired_text, "compliances": compliance}
                results.append(result)
                break

            # collect retry results
            result = {
                "response": repaired_text,
                "compliances": compliance,
            }
            results.append(result)

        # pick the best repaired response
        repaired_text = self.select_bestofn(results)

        return PolicyRepairerOutput(
            repaired_text=repaired_text, bestofn_attempts=results
        )


class MapReduceRepairer(Repairer):
    name: str = MAPREDUCE_REPAIR_NAME
    surrogaterepairer: Repairer  # = BestofNRepairer(detector=detector, model_id=model_id, model_provider=model_provider)

    def repair(self, repair_input: PolicyRepairerInput) -> PolicyRepairerOutput:
        policies = repair_input.detection_input.policies
        detector_inputs = PolicyDetectorInput(
            policies=policies,
            prompt=repair_input.detection_input.prompt,
            response=repair_input.detection_input.response,
        )
        detection = self.detector.detect(detector_inputs)
        compliance = [d.compliance for d in detection.policy_outputs]

        # single_instruction_repair_results = []
        single_instruction_responses = []
        for i in range(len(policies)):
            if not compliance[i]:
                new_inputs = copy.deepcopy(repair_input.detection_input)
                new_inputs.policies = [policies[i]]
                retn = self.surrogaterepairer.repair(
                    inputs=new_inputs, repair_inputs=repair_input
                )
                # single_instruction_repair_results.append(retn)
                single_instruction_responses.append(retn.repaired_text)
            else:
                single_instruction_responses.append(
                    repair_input.detection_input.response
                )

        prompt = mapreduce_repair_prompt(
            model_id=self.config.llm_client.model_id,
            policies=policies,
            responses=single_instruction_responses,
            query=repair_input.detection_input.prompt,
        )
        repaired_text = self.repair_endpoint(
            prompt, self.config.llm_client.model_id, return_counts=False
        )
        repaired_text = self.postprocess_repair_text(repaired_text)

        return PolicyRepairerOutput(repaired_text=repaired_text)
