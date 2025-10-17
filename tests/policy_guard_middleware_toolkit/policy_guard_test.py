from dotenv import load_dotenv
import os

import pytest
from altk.core.llm.base import get_llm
from altk.core.toolkit import AgentPhase, ComponentConfig
from altk.pre_response.policy_guard.core.toolkit import (
    PolicyDetectorInput,
    PolicyRepairerInput,
)
from altk.pre_response.policy_guard.detect.detector import (
    SinglePolicyLLMDetector,
    BatchPolicyLLMDetector,
)
from altk.pre_response.policy_guard.repair.repairer import (
    BatchPolicyRepairer,
    BestofNRepairer,
    BestofNGenerator,
    IterativeRepairer,
    RetryRepairer,
)


load_dotenv()


class TestPolicyGuard:
    @pytest.fixture
    def llm_client(self):
        WATSONX_CLIENT = get_llm("watsonx")
        llm_client = WATSONX_CLIENT(
            model_id="meta-llama/llama-3-3-70b-instruct",
            api_key=os.getenv("WX_API_KEY"),
            project_id=os.getenv("WX_PROJECT_ID"),
            url=os.getenv("WX_URL", "https://us-south.ml.cloud.ibm.com"),
        )
        yield llm_client

    def test_detect(self, llm_client):
        config = ComponentConfig(llm_client=llm_client)

        detector = BatchPolicyLLMDetector(config=config)

        query = "What is a good national park to visit for hiking?"
        policies = [
            "Don't talk about national parks in California",
            "Make sure to include at least one national park in New York",
            "Talk about the wildlife in the park",
            "Remind users to bring bugspray",
        ]
        response = "A great national park to visit for hiking is Glacier National Park in Montana, \
            known for its stunning alpine scenery, dramatic peaks, and over 700 miles of trails ranging from easy walks to \
                challenging backcountry routes. Hikers can explore glacier-carved valleys, turquoise lakes, \
                    and abundant wildlife, including mountain goats and bears. The iconic Highline Trail offers \
                        breathtaking views along the Garden Wall, while the Grinnell Glacier Trail provides a rewarding ascent to \
                            one of the park’s most famous glaciers. With its diverse terrain and relatively uncrowded feel \
                                compared to other major parks, Glacier is a top destination for avid hikers. Don't forget to bring your bugspray!"

        input_data = PolicyDetectorInput(
            prompt=query, policies=policies, response=response
        )
        result = detector.process(input_data, AgentPhase.RUNTIME)

        assert len(result.policy_outputs) == 4
        expected_results = [True, False, True, True]
        for policy, expected, output in zip(
            policies, expected_results, result.policy_outputs
        ):
            assert output.compliance == expected, (
                f"Poicy: {policy} -> expected compliance {expected} didn't match result"
            )

    # def test_messages(self, tmp_path, llm_client):
    #     config = ComponentConfig(llm_client=llm_client)

    #     detector = BatchPolicyLLMDetector()

    #     query = "What is a good national park to visit for hiking?"
    #     policies = [
    #         "Don't talk about national parks in California",
    #         "Make sure to include at least one national park in New York",
    #         "Talk about the wildlife in the park",
    #         "Remind users to bring bugspray"
    #     ]
    #     response = "A great national park to visit for hiking is Glacier National Park in Montana, \
    #         known for its stunning alpine scenery, dramatic peaks, and over 700 miles of trails ranging from easy walks to \
    #             challenging backcountry routes. Hikers can explore glacier-carved valleys, turquoise lakes, \
    #                 and abundant wildlife, including mountain goats and bears. The iconic Highline Trail offers \
    #                     breathtaking views along the Garden Wall, while the Grinnell Glacier Trail provides a rewarding ascent to \
    #                         one of the park’s most famous glaciers. With its diverse terrain and relatively uncrowded feel \
    #                             compared to other major parks, Glacier is a top destination for avid hikers. Don't forget to bring your bugspray!"
    #     messages=[
    #         HumanMessage(content=query),
    #         AIMessage(content=response)
    #     ]
    #     input_data = PolicyDetectorInput(
    #         messages=messages,
    #         policies=policies
    #     )
    #     result = detector.process(input_data, AgentPhase.RUNTIME)
    #     assert result is not None

    def test_repair(self, llm_client):
        config = ComponentConfig(llm_client=llm_client)

        detector = BatchPolicyLLMDetector(config=config)
        repairer = BatchPolicyRepairer(config=config, detector=detector)

        query = "What is a good national park to visit for hiking?"
        policies = [
            "Don't talk about national parks in California",
            "Make sure to include at least one national park in New York",
            "Talk about the wildlife in the park",
            "Remind users to bring bugspray",
        ]
        response = "A great national park to visit for hiking is Glacier National Park in Montana, \
            known for its stunning alpine scenery, dramatic peaks, and over 700 miles of trails ranging from easy walks to \
                challenging backcountry routes. Hikers can explore glacier-carved valleys, turquoise lakes, \
                    and abundant wildlife, including mountain goats and bears. The iconic Highline Trail offers \
                        breathtaking views along the Garden Wall, while the Grinnell Glacier Trail provides a rewarding ascent to \
                            one of the park’s most famous glaciers. With its diverse terrain and relatively uncrowded feel \
                                compared to other major parks, Glacier is a top destination for avid hikers. Don't forget to bring your bugspray!"

        input_data = PolicyDetectorInput(
            prompt=query, policies=policies, response=response
        )
        result = detector.process(input_data, AgentPhase.RUNTIME)
        repair_input = PolicyRepairerInput(
            detection_input=input_data, detection_output=result
        )
        repair_result = repairer.process(repair_input, AgentPhase.RUNTIME)

        assert repair_result is not None

    def test_component_init(self, llm_client):
        # TODO: Check whether this this test should be removed or not?
        config = ComponentConfig(llm_client=llm_client)

        detector_1 = BatchPolicyLLMDetector(config=config)
        SinglePolicyLLMDetector(config=config)
        BatchPolicyRepairer(config=config, detector=detector_1)
        IterativeRepairer(config=config, detector=detector_1)
        RetryRepairer(config=config, detector=detector_1)
        BestofNRepairer(config=config, detector=detector_1)
        BestofNGenerator(config=config, detector=detector_1)
