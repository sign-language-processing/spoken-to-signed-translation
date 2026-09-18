# Gloss-Based Pipeline for Spoken to Signed Language Translation

a `text-to-gloss-to-pose-to-video` pipeline for spoken to signed language translation.

- Demos available for:
  - 🇩🇪 [Swiss German Sign Language](https://sign.mt/?sil=sgg&spl=de) 🇨🇭
  - 🇫🇷 [French Sign Language of Switzerland](https://sign.mt/?sil=ssr&spl=fr)🇨🇭
  - 🇮🇹 [Italian Sign Language of Switzerland](https://sign.mt/?sil=slf&spl=it) 🇨🇭

- Paper available on [arxiv](https://arxiv.org/abs/2305.17714), presented
  at [AT4SSL 2023](https://sites.google.com/tilburguniversity.edu/at4ssl2023/).

![Visualization of our pipeline](assets/pipeline.jpg)

## Install

```bash
pip install spoken-to-signed
```

Then, to download a lexicon, run:
```bash
download_lexicon \
  --name <signsuisse> \
  --directory <path_to_directory>
```

## Usage

For language codes, we use the [IANA Language Subtag Registry](https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry).
Our pipeline provides multiple scripts.

To quickly demo it using a dummy lexicon, either open it in Colab:

<a target="_blank" href="https://colab.research.google.com/drive/1UtBmfBIhUa2EdLMnWJr0hxAOZelQ50_9?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

or run it locally with some installation steps first:

```bash
git clone https://github.com/ZurichNLP/spoken-to-signed-translation
cd spoken-to-signed-translation

pip install .

text_to_gloss_to_pose \
  --text "Kleine Kinder essen Pizza in Zürich." \
  --glosser "simple" \
  --lexicon "assets/dummy_lexicon" \
  --spoken-language "de" \
  --signed-language "sgg" \
  --pose "quick_test.pose"
```

#### Text-to-Gloss Translation

This script translates input text into gloss notation.

```bash
text_to_gloss \
  --text <input_text> \
  --glosser <simple|spacylemma|rules|nmt> \
  --spoken-language <de|fr|it> \
  --signed-language <sgg|ssr|slf>
```

#### Text-to-Gloss-to-Pose Translation

This script translates input text into gloss notation, then converts the glosses into a pose file.

```bash
text_to_gloss_to_pose \
  --text <input_text> \
  --glosser <simple|spacylemma|rules|nmt> \
  --lexicon <path_to_directory> \
  --spoken-language <de|fr|it> \
  --signed-language <sgg|ssr|slf> \
  --pose <output_pose_file_path>.pose
```

Words missing from the lexicon are fingerspelled letter-by-letter by default;
pass `--disable-fingerspelling` to skip them instead.

Add `--coverage-info` to print how each token was matched, color-coded in the terminal
(green: lexicon, yellow: language backup, orange: fingerspelling, red: unmatched),
or `--coverage-stats <file.json>` to save the same per-token information as JSON.

#### Text-to-Gloss-to-Pose-to-Video Translation

This script translates input text into gloss notation, converts the glosses into a pose file, and then transforms the pose file into a video.

> **Note:** Video generation requires the `pose-to-video` package with pix2pix and upscaler:
> ```bash
> pip install 'pose-to-video[pix2pix,simple_upscaler] @ git+https://github.com/sign-language-processing/pose-to-video'
> ```

```bash
text_to_gloss_to_pose_to_video \
  --text <input_text> \
  --glosser <simple|spacylemma|rules|nmt> \
  --lexicon <path_to_directory> \
  --spoken-language <de|fr|it> \
  --signed-language <sgg|ssr|slf> \
  --video <output_video_file_path>.mp4
```

## HTTP service

```bash
pip install '.[server]'
MODEL_VERSION=local hypercorn spoken_to_signed.server:app --bind 0.0.0.0:8080
```

`POST /tokens-to-gloss` wraps the existing `rules` (default, English → ASL) or
`simple` (identity) token glosser. It does not tokenize, look up signs, or construct poses.
Send atomic words or multiword spans, with POS and optional morphology from WSD:

```json
{
  "spoken_language": "en",
  "signed_language": "ase",
  "tokens": [
    {"word": "What", "gloss": "what", "pos": "PRON"},
    {"word": "is", "gloss": "be", "pos": "AUX"},
    {"word": "your", "gloss": "your", "pos": "PRON"},
    {"word": "name", "gloss": "name", "pos": "NOUN"},
    {"word": "?", "gloss": "?", "pos": "PUNCT"}
  ]
}
```

Response: `{"sentences": [[2, 3, 0, 4]]}` — **your name what ?**.
Indexes refer to the input items, not raw WSD token positions. The caller retains each
item's senses, entity links, and source spans. Missing indexes are dropped; unknown
words remain available for downstream fingerspelling. Punctuation remains for sentence
boundaries, not dictionary lookup. `morphology` is a list of spaCy feature dictionaries,
one per source token in an item. The rules are a mechanical baseline, not fluent ASL.

`GET /health` returns `version`; successful responses include `X-Model-Tag`.
Set `MODEL_VERSION` to identify the deployed build for downstream caches. `/docs`
documents the request schema. Each request handles one document; TODO: batch API.

```bash
docker build --build-arg MODEL_VERSION=local -t spoken-to-signed .
docker run --rm -p 8080:8080 spoken-to-signed
```

GitHub Actions builds the image on PRs and publishes
`ghcr.io/sign-language-processing/spoken-to-signed-translation:<release-tag>` on releases,
with a unique build version baked in. The CPU image needs no spaCy model or database;
`PORT` defaults to `8080`. Authentication and caching belong to the calling gateway;
deploy this service on an internal network.

Hypercorn serves HTTP/1.1 and cleartext HTTP/2 (h2c) on the same port. The container
smoke test checks both. Configure the gateway/deployment to use HTTP/2 upstream;
server support alone does not make every connection HTTP/2.

### Optional pose lookup

`POST /gloss-to-pose` takes **already ordered** `tokens`, `spoken_language`, and
`signed_language`, and returns binary `application/pose`. It delegates lookup,
fingerspelling, and concatenation to the existing library. Punctuation marked
`pos: "PUNCT"` is excluded. Optional parameters: `fingerspelling` (default `true`),
`anonymize` (default `false`), and `source` (PostgreSQL video-ID prefix filter).

Configure one server-side backend:

- `LEXICON_PATH`: a local CSV lexicon directory.
- `DATABASE_URL`: PostgreSQL connection string (takes precedence). Install
  `.[server,postgres,gcs]`; these extras are included in the Docker image.

The PostgreSQL backend is migrated from `models/functions_py/spoken_text_to_signed_pose`.
It queries the existing `captions` table (`videoId`, `language`, `videoLanguage`,
`start`, `end`, `text`, `lemmas`), then reads public poses from `gs://sign-mt-poses`.
This is the legacy caption lookup, **not** synset/entity lookup through dictionary-api.
Use a read-only database role. No database is contacted by health/glossing requests.
Without a configured backend, pose requests return 503.

Library users can also supply this backend directly:

```python
from spoken_to_signed.gloss_to_pose.lookup.sql_lookup import SQLPoseLookup
from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup

lookup = SQLPoseLookup(database_config={"dsn": database_url}, backup=FingerspellingPoseLookup())
```

Omit `backup` to disable fingerspelling. Candidate lookup is one query per gloss
sequence; the existing lookup implements language fallback and coverage reporting.
The HTTP service does not copy the old Firebase scheduler, interpreter asset, browser
authentication, or mutable global request settings. Migrate callers and the caption
maintenance job separately before retiring the old function in `models`.

## Methodology

The pipeline consists of three main components:

1. **Text-to-Gloss Translation**

   Transforms the input (spoken language) text into a sequence of glosses.

  - [Simple lemmatizer](spoken_to_signed/text_to_gloss/simple.py),
  - [Spacy lemmatizer: more accurate, but slower lemmatization, covering fewer languages than `simple`](spoken_to_signed/text_to_gloss/spacylemma.py),
  - [Rule-based word reordering and dropping](spoken_to_signed/text_to_gloss/rules.py) component and
  - [Neural machine translation system](spoken_to_signed/text_to_gloss/nmt.py).

2. **Gloss-to-Pose Conversion**

  - [Lookup](spoken_to_signed/gloss_to_pose/lookup/lookup.py): Uses a lexicon of signed languages to convert the sequence of glosses into a
      sequence of poses.
  - [Pose Concatenation](spoken_to_signed/gloss_to_pose/concatenate.py): The poses are then cropped, concatenated, and smoothed,
      creating a pose representation for the input sentence.

3. **Pose-to-Video Generation**

    Transforms the processed pose video back into a synthesized video using an image translation model.

## Supported Languages

| Language                    | IANA Code | Glossers Supported                                                                                                                                         | Lexicon Data Source                                  |
|-----------------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| Swiss German Sign Language  | sgg       | `simple`, `spacylemma`, `rules`, [`nmt`](https://github.com/ZurichNLP/spoken-to-signed-translation/tree/main/spoken_to_signed/text_to_gloss#nmt-component) | [SignSuisse (de)](https://signsuisse.sgb-fss.ch/de/) |
| Swiss French Sign Language  | ssr       | `simple`, `spacylemma`                                                                                                                                                   | [SignSuisse (fr)](https://signsuisse.sgb-fss.ch/fr/) |
| Swiss Italian Sign Language | slf       | `simple`, `spacylemma`                                                                                                                                                   | [SignSuisse (it)](https://signsuisse.sgb-fss.ch/it/) |
| German Sign Language        | gsg       | `simple`, `spacylemma`, [`nmt`](https://github.com/ZurichNLP/spoken-to-signed-translation/tree/main/spoken_to_signed/text_to_gloss#nmt-component)                        | WordNet (Coming Soon)                                |
| British Sign Language       | bfi       | `simple`, `spacylemma`, [`nmt`](TODO-model-link)                                                                                                                         | WordNet (Coming Soon)                                |

## Online Playgrounds

We have two available:

- [sign.mt](https://sign.mt) is a web interface of a translation system.
- [research.sign.mt](https://research.sign.mt) is an overview of sign language processing literature.

## Citation

If you find this work useful, please cite our paper:

```bib
@inproceedings{moryossef2023baseline,
  title={An Open-Source Gloss-Based Baseline for Spoken to Signed Language Translation},
  author={Moryossef, Amit and M{\"u}ller, Mathias and G{\"o}hring, Anne and Jiang, Zifan and Goldberg, Yoav and Ebling, Sarah},
  booktitle={2nd International Workshop on Automatic Translation for Signed and Spoken Languages (AT4SSL)},
  year={2023},
  month={June},
  url={https://github.com/ZurichNLP/spoken-to-signed-translation},
  note={Available at: \url{https://arxiv.org/abs/2305.17714}}
}
```
