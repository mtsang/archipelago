from transformers import Pipeline
from typing import Dict, List, Optional, Tuple, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.modelcard import ModelCard
from transformers.tokenization_auto import AutoTokenizer
from transformers.configuration_auto import (
    ALL_PRETRAINED_CONFIG_ARCHIVE_MAP,
    AutoConfig,
)
import torch

from transformers.modeling_auto import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
)


class TextClassificationPipeline(Pipeline):
    """
    Text classification pipeline using ModelForTextClassification head.
    """

    def __call__(self, *args, **kwargs):
        outputs = super().__call__(*args, **kwargs)
        #         scores = np.exp(outputs) / np.exp(outputs).sum(-1)
        return outputs  # [{"label": self.model.config.id2label[item.argmax()], "score": item.max()} for item in scores]


# Register all the supported task here
SUPPORTED_TASKS = {
    "sentiment-analysis": {
        "impl": TextClassificationPipeline,
        "pt": AutoModelForSequenceClassification,  # if is_torch_available() else None,
        "default": {
            "model": {
                "pt": "distilbert-base-uncased-finetuned-sst-2-english",
            },
            "config": "distilbert-base-uncased-finetuned-sst-2-english",
            "tokenizer": "distilbert-base-uncased",
        },
    },
}


def pipeline(
    task: str,
    model: Optional = None,
    config: Optional[Union[str, PretrainedConfig]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    modelcard: Optional[Union[str, ModelCard]] = None,
    device=torch.device("cpu"),
    **kwargs
) -> Pipeline:
    """
    Utility factory method to build a pipeline.
    Pipeline are made of:
        A Tokenizer instance in charge of mapping raw textual input to token
        A Model instance
        Some (optional) post processing for enhancing model's output
    Examples:
        pipeline('sentiment-analysis')
        pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', tokenizer='bert-base-cased')
        pipeline('ner', model=AutoModel.from_pretrained(...), tokenizer=AutoTokenizer.from_pretrained(...)
        pipeline('ner', model='dbmdz/bert-large-cased-finetuned-conll03-english', tokenizer='bert-base-cased')
        pipeline('ner', model='https://...pytorch-model.bin', config='https://...config.json', tokenizer='bert-base-cased')
    """
    # Retrieve the task
    if task not in SUPPORTED_TASKS:
        raise KeyError(
            "Unknown task {}, available tasks are {}".format(
                task, list(SUPPORTED_TASKS.keys())
            )
        )

    framework = "pt"  # get_framework(model)

    targeted_task = SUPPORTED_TASKS[task]
    task, model_class = targeted_task["impl"], targeted_task[framework]

    # Use default model/config/tokenizer for the task if no model is provided
    if model is None:
        models, config, tokenizer = tuple(targeted_task["default"].values())
        model = models[framework]

    # Try to infer tokenizer from model or config name (if provided as str)
    if tokenizer is None:
        if isinstance(model, str) and model in ALL_PRETRAINED_CONFIG_ARCHIVE_MAP:
            tokenizer = model
        elif isinstance(config, str) and config in ALL_PRETRAINED_CONFIG_ARCHIVE_MAP:
            tokenizer = config
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Impossible to guess which tokenizer to use. "
                "Please provided a PretrainedTokenizer class or a path/url/shortcut name to a pretrained tokenizer."
            )

    # Try to infer modelcard from model or config name (if provided as str)
    if modelcard is None:
        # Try to fallback on one of the provided string for model or config (will replace the suffix)
        if isinstance(model, str):
            modelcard = model
        elif isinstance(config, str):
            modelcard = config

    # Instantiate tokenizer if needed
    if isinstance(tokenizer, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Instantiate config if needed
    if isinstance(config, str):
        config = AutoConfig.from_pretrained(config)

    # Instantiate modelcard if needed
    if isinstance(modelcard, str):
        modelcard = ModelCard.from_pretrained(modelcard)

    # Instantiate model if needed
    if isinstance(model, str):
        # Handle transparent TF/PT model conversion
        model_kwargs = {}
        if framework == "pt" and model.endswith(".h5"):
            model_kwargs["from_tf"] = True
            logger.warning(
                "Model might be a TensorFlow model (ending with `.h5`) but TensorFlow is not available. "
                "Trying to load the model with PyTorch."
            )
        #         else:
        #             raise ValueError("invalid framework or model type")

        model = model_class.from_pretrained(model, config=config, **model_kwargs)
        model = model.to(device)

    return task(
        model=model,
        tokenizer=tokenizer,
        modelcard=modelcard,
        framework=framework,
        **kwargs
    )
