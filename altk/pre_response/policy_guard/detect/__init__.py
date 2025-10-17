from altk.pre_response.policy_guard.detect.detector import (
    Detector,
    SinglePolicyLLMDetector,
    BatchPolicyLLMDetector,
    SINGLE_DETECTOR_NAME,
    BATCH_DETECTOR_NAME,
)


# Factory method
def detector_factory(detector_type: str, config) -> Detector:
    if detector_type == SINGLE_DETECTOR_NAME:
        return SinglePolicyLLMDetector(name=detector_type, config=config)
    elif detector_type == BATCH_DETECTOR_NAME:
        return BatchPolicyLLMDetector(name=detector_type, config=config)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
