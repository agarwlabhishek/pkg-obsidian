import os
import json
import requests
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from temporalio import activity
from .di_text_extraction import DocumentExtractor
from .text_anonymization import TextAnonymizer
from .email_preprocessing import EmailPreprocessor
from .medical_provider_ranking import ProviderInfo, ProviderRanker


@dataclass
class DocumentExtract:
    file_path: Union[str, Path]
    force_model: Optional[str] = None
    auto_detect_model: Optional[bool] = True


@dataclass
class AnonymizationText:
    text: str
    language: str
    custom_operators_config: Optional[Dict[str, Dict[str, Any]]] = None


@dataclass
class EmailPreprocess:
    msg_path: str


@dataclass
class ProviderRanking:
    provider_info: Dict[str, Any]
    provider_candidates: List[Dict[str, Any]]
    top_n: int
    common_term_thershold: Optional[float] = 0.10


@activity.defn
async def extract_text_from_document(request: Dict[str, Any]) -> Dict[str, Any]:
    typed_request = DocumentExtract(**request)
    extractor = DocumentExtractor()
    result = extractor.extract_text(typed_request.file_path)
    di_result = {
        "summary": {
            "file": result.metadata["file_name"],
            "model_used": result.model_used,
            "metadata": result.metadata,
            "detected_content": result.text_content,
            "key_value_pairs": result.key_value_pairs,
        },
        "metadata": result.metadata,
    }
    return di_result


@activity.defn
async def anonymize_text(request: Dict[str, Any]) -> Dict[str, Any]:
    typed_request = AnonymizationText(**request)
    anonymizer = TextAnonymizer(language=typed_request.language)
    custom_config = None
    if typed_request.custom_operators_config:
        custom_config = anonymizer.create_custom_operators_config(typed_request.custom_operators_config)
    result = anonymizer.anonymize(typed_request.text, anonymization_config=custom_config)
    return asdict(result)


@activity.defn
async def email_preprocessing(request: Dict[str, Any]) -> Dict[str, Any]:
    typed_request = EmailPreprocess(**request)
    preprocessor = EmailPreprocessor(preserve_temp_files=False)
    email_data = preprocessor.process_msg_file(typed_request.msg_file_path)
    return asdict(email_data)


@activity.defn
async def medical_provider_ranking(request: Dict[str, Any]) -> Dict[str, Any]:
    typed_request = ProviderRanking(**request)
    provider_info = ProviderInfo(**typed_request.provider_info)
    ranker = ProviderRanker()
    ranker.similarity_calculator._common_terms_threshold = typed_request.common_term_thershold
    results = ranker.rank_providers(provider_info, typed_request.provider_candidates, top_n=typed_request.top_n)
    return asdict(results)


## Common Activities


@activity.defn
async def delete_request_files(workflow_id: str, files: List[str], root_path: str = "/mnt/artifacts"):
    path = f"{root_path}/{workflow_id}"
    for file in files:
        if os.path.exists(f"{path}/{file}"):
            os.remove(f"{path}/{file}")
        else:
            raise Exception(f"{file} does not exist in the specified path.")


@activity.defn
async def validate_file_exist(workflow_id: str, files: List[str], root_path: str = "/mnt/artifacts"):
    path = f"{root_path}/{workflow_id}"
    for file in files:
        if not os.path.exists(f"{path}/{file}"):
            raise Exception(f"{file} does not exist in the specified path.")


@activity.defn
async def api_request(
    url: str, method: str, body: Any = None, isBodyJson: bool = False, headers: Dict[str, Any] = None
):
    if method.upper() not in ["GET", "PUT", "POST", "DELETE"]:
        raise Exception(f"Method provided: {method} does not have any of the allowed values (GET, PUT, POST, DELETE)")
    if body is not None and isinstance(body, str) and isBodyJson:
        json.loads(body)
    elif body is not None and isinstance(body, Dict):
        pass
    elif body is None:
        pass
    else:
        raise Exception("Provided body does not match the expected type.")
    if isBodyJson:
        response = requests.request(method.upper(), url, headers=headers, json=body)
    else:
        response = requests.request(method.upper(), url, headers=headers, data=body)
    return json.loads(response)
