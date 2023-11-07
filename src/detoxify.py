# This code was taken from Detoxify library and tweaked to work with CBA infrastructure (https://github.com/unitaryai/detoxify)

import torch
import transformers
from typing import Tuple, Any

PRETRAINED_MODEL = None


def get_model_and_tokenizer(
    burt_model_loc: str,
    model_type: str,
    model_name: str,
    tokenizer_name: str,
    num_classes: int,
    state_dict: dict,
) -> Tuple[Any, Any]:
    burt_model_location = burt_model_loc + model_type
    model_class = getattr(transformers, model_name)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=burt_model_location,
        num_labels=num_classes,
        state_dict=state_dict,
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(
        burt_model_loc + "roberta-base"
    )
    return model, tokenizer


def load_checkpoint(
    burt_model_loc: str,
    model_type: str = "original",
    checkpoint: str = None,
    device: str = "cpu",
) -> Tuple[Any, Any, str]:
    if checkpoint is None:
        checkpoint_path = model_type
        loaded = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location=device
        )
    else:
        loaded = torch.load(checkpoint)
        if "config" not in loaded or "state_dict" not in loaded:
            raise ValueError(
                "Checkpoint needs to contain the config it was trained \
                    with as well as the state dict"
            )
    class_names = loaded["config"]["dataset"]["args"]["classes"]

    model, tokenizer = get_model_and_tokenizer(
        burt_model_loc,
        **loaded["config"]["arch"]["args"],
        state_dict=loaded["state_dict"]
    )
    return model, tokenizer, class_names


def load_model(model_type: str, checkpoint: str = None) -> Any:
    if checkpoint is None:
        model, _, _ = load_checkpoint(model_type=model_type)
    else:
        model, _, _ = load_checkpoint(checkpoint=checkpoint)
    return model


class Detoxify:
    """Detoxify
    Easily predict if a comment or list of comments is toxic.
    Can initialize 3 different model types from model type or checkpoint path:
        - unbiased:
            model trained on data from the Jigsaw Unintended Bias in
            Toxicity Classification Challenge
    Args:
        model_type(str): model type to be loaded, can be either original,
                         unbiased or multilingual
        checkpoint(str): checkpoint path, defaults to None
        device(str or torch.device): accepts any torch.device input or
                                     torch.device object, defaults to cpu
    Returns:
        results(dict): dictionary of output scores for each class
    """

    def __init__(
        self,
        burt_model_loc: str,
        model_type: str = "unbiased",
        checkpoint: str = PRETRAINED_MODEL,
        device: str = "cpu",
    ) -> None:
        super(Detoxify, self).__init__()
        if burt_model_loc[-1] != "/":
            burt_model_loc = burt_model_loc + "/"
        self.model, self.tokenizer, self.class_names = load_checkpoint(
            burt_model_loc,
            model_type=model_type,
            checkpoint=checkpoint,
            device=device,
        )
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        self.model.eval()
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.model.device)
        out = self.model(**inputs)[0]
        scores = torch.sigmoid(out).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(self.class_names):
            results[cla] = (
                scores[0][i]
                if isinstance(text, str)
                else [scores[ex_i][i].tolist() for ex_i in range(len(scores))]
            )
        return results
