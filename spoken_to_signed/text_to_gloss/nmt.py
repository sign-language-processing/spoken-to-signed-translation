import os
import tarfile
import requests
import torch as pt
import sentencepiece as spm
import sockeye.inference

from sockeye import inference, model
from typing import Dict, List, Any

from .types import Gloss


MODELS_PATH = './models'


def download_and_extract_file(url: str, filepath: str):

    print("Attempting to download and extract: %s" % url)

    r = requests.get(url)

    filepath_tar_ball = filepath + ".tar.gz"

    open(filepath_tar_ball, 'wb').write(r.content)

    tar = tarfile.open(filepath_tar_ball)
    tar.extractall(path=MODELS_PATH)
    tar.close()

    print("Model saved to : %s" % filepath)


def download_model_if_does_not_exist(sockeye_paths: Dict[str, str]):

    model_path = sockeye_paths["model_path"]
    url = sockeye_paths["url"]

    if not os.path.exists(model_path):
        download_and_extract_file(url, model_path)

    assert os.path.exists(model_path), "Model folder '%s' does not exist after " \
                                       "attempting to download and extract." % model_path


def load_sockeye_models():

    os.makedirs(MODELS_PATH, exist_ok=True)

    spm_name = "sentencepiece.model"

    sockeye_paths_dict = {
        "dgs_de": {
                         "model_path": os.path.join(MODELS_PATH, "dgs_de"),
                         "spm_path": os.path.join(MODELS_PATH, "dgs_de", spm_name),
                         "url": "https://files.ifi.uzh.ch/cl/archiv/2022/easier/dgs_de.tar.gz"
                        }
    }

    sockeye_models_dict = {}

    device = pt.device('cpu')

    for model_name in sockeye_paths_dict.keys():

        sockeye_paths = sockeye_paths_dict[model_name]

        download_model_if_does_not_exist(sockeye_paths)

        model_path = sockeye_paths["model_path"]
        spm_path = sockeye_paths["spm_path"]

        sockeye_models, sockeye_source_vocabs, sockeye_target_vocabs = model.load_models(
            device=device, dtype=None, model_folders=[model_path], inference_only=True)

        sockeye_models_dict[model_name] = {"sockeye_models": sockeye_models,
                                           "spm_model": spm.SentencePieceProcessor(model_file=spm_path),
                                           "sockeye_source_vocabs": sockeye_source_vocabs,
                                           "sockeye_target_vocabs": sockeye_target_vocabs}

    return device, sockeye_paths_dict, sockeye_models_dict


device, sockeye_paths_dict, sockeye_models_dict = load_sockeye_models()


def apply_pieces(text: str, spm_model: spm.SentencePieceProcessor) -> str:
    text = text.strip()

    pieces = spm_model.encode(text, out_type=str)

    return " ".join(pieces)


def remove_pieces(translation: str) -> str:
    """

    :param translation:
    :return:
    """
    translation = translation.replace(" ", "")
    translation = translation.replace("▁", " ")

    return translation.strip()


def add_tag_to_text(text: str, tag: str) -> str:
    text = text.strip()

    if text == "":
        return ""

    tokens = text.split(" ")
    tokens = [tag] + tokens

    return " ".join(tokens)


def translate(text: str,
              source_language_code: str = "de",
              target_language_code: str = "dgs",
              nbest_size: int = 3) -> Dict[str, Any]:

    if source_language_code == "de":
        model_name = "dgs_de"
    else:
        raise NotImplementedError()

    sockeye_models = sockeye_models_dict[model_name]["sockeye_models"]
    sockeye_source_vocabs = sockeye_models_dict[model_name]["sockeye_source_vocabs"]
    sockeye_target_vocabs = sockeye_models_dict[model_name]["sockeye_target_vocabs"]

    spm_model = sockeye_models_dict[model_name]["spm_model"]

    pieces = apply_pieces(text, spm_model)

    tag_str = '<2{}>'.format(target_language_code)
    tagged_pieces = add_tag_to_text(pieces, tag_str)

    beam_size = nbest_size

    translator = inference.Translator(device=device,
                                      ensemble_mode='linear',
                                      scorer=inference.CandidateScorer(),
                                      output_scores=True,
                                      batch_size=1,
                                      beam_size=beam_size,
                                      beam_search_stop='all',
                                      nbest_size=nbest_size,
                                      models=sockeye_models,
                                      source_vocabs=sockeye_source_vocabs,
                                      target_vocabs=sockeye_target_vocabs)

    input_ = inference.make_input_from_plain_string(0, tagged_pieces)
    output = translator.translate([input_])[0]  # type: sockeye.inference.TranslatorOutput

    translations = output.nbest_translations  # type: List[str]
    translations = [remove_pieces(t) for t in translations]

    return {
        'source_language_code': source_language_code,
        'target_language_code': target_language_code,
        'nbest_size': nbest_size,
        'text': text,
        'translations': translations,
    }


def text_to_gloss(text: str, language: str, nbest_size: int = 3) -> List[Gloss]:
    if language == "de":

        translations_dict = translate(text=text,
                                      source_language_code="de",
                                      target_language_code="dgs",
                                      nbest_size=nbest_size)
    else:
        raise NotImplementedError()

    best_translation = translations_dict["translations"][0]  # type: str
    glosses = best_translation.split(" ")

    tokens = [None] * len(glosses)

    return [list(zip(tokens, glosses))]
