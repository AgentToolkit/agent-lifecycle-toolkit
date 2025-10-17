from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams


def generate_params(
    max_new_tokens=1500, temperature=0.0, decoding_method="greedy"
) -> dict:
    params = {
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.TEMPERATURE: temperature,
    }
    return params


model_id_mistral = "mistralai/mistral-large"
model_id_llama = "meta-llama/llama-3-3-70b-instruct"
model_id_llama_8b = "meta-llama/Llama-3.1-8B-Instruct"
model_id_granite = "ibm/granite-3-2-8b-instruct"
model_id_mixtral = "mistralai/mixtral-8x22B-instruct-v0.1"
model_id_mixtral7b = "mistralai/mixtral-8x7B-instruct-v0.1"
model_id_qwen2 = "Qwen/Qwen2.5-72B-Instruct"
model_id_qwen3 = "Qwen/Qwen3-8B"
model_id_gpt4o = "gpt-4o-2024-08-06"


def get_model_name(model_id: str) -> str:
    if model_id == model_id_mistral:
        return "mistral"
    elif model_id == model_id_granite:
        return "granite"
    elif model_id == model_id_llama:
        return "llama"
    elif model_id == model_id_llama_8b:
        return "llama8b"
    elif model_id == model_id_mixtral:
        return "mixtral"
    elif model_id == model_id_mixtral7b:
        return "mixtral7b"
    elif model_id == model_id_qwen2:
        return "qwen2"
    elif model_id == model_id_qwen3:
        return "qwen3"
    elif model_id == model_id_gpt4o:
        return "gpt4o"
    else:
        raise Exception(f"Unknown model_id {model_id}")
