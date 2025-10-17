from altk.core.toolkit import ComponentConfig
from altk.pre_response.policy_guard.detect.detector import Detector
from altk.pre_response.policy_guard.repair.repairer import (
    Repairer,
    IterativeRepairer,
    BatchPolicyRepairer,
    RetryRepairer,
    BestofNRepairer,
    BestofNGenerator,
    MapReduceRepairer,
    BATCH_REPAIR_NAME,
    ITERATIVE_REPAIR_NAME,
    RETRY_REPAIR_NAME,
    BESTOFN_REPAIR_NAME,
    BESTOFNGEN_REPAIR_NAME,
    MAPREDUCE_REPAIR_NAME,
)


# Factory method
def repairer_factory(
    repairer_type: str, detector: Detector, config: ComponentConfig
) -> Repairer:
    if repairer_type == BATCH_REPAIR_NAME:
        return BatchPolicyRepairer(name=repairer_type, detector=detector, config=config)
    elif repairer_type == ITERATIVE_REPAIR_NAME:
        return IterativeRepairer(name=repairer_type, detector=detector, config=config)
    elif repairer_type == RETRY_REPAIR_NAME:
        return RetryRepairer(name=repairer_type, detector=detector, config=config)
    elif repairer_type == BESTOFN_REPAIR_NAME:
        return BestofNRepairer(name=repairer_type, detector=detector, config=config)
    elif repairer_type == BESTOFNGEN_REPAIR_NAME:
        return BestofNGenerator(name=repairer_type, detector=detector, config=config)
    elif repairer_type == MAPREDUCE_REPAIR_NAME:
        raise Exception("Mapreduce repairer not implemented")
        return MapReduceRepairer(name=repairer_type, detector=detector, config=config)
    else:
        raise ValueError(f"Unknown repairer type: {repairer_type}")
