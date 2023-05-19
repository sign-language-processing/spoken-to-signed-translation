# Gloss-Based Pipeline for Spoken to Signed Language Translation

a `text-to-gloss-to-pose-to-video` pipeline for spoken to signed language translation.

- Demos available for:
  - 🇩🇪 [Swiss German Sign Language](https://sign.mt/?sil=ch&spl=de) 🇨🇭
  - 🇫🇷 [French Sign Language of Switzerland](https://sign.mt/?sil=ch&spl=fr)🇨🇭
  - 🇮🇹 [Italian Sign Language of Switzerland](https://sign.mt/?sil=ch&spl=it) 🇨🇭

- Paper available on [arxiv](https://arxiv.org/abs/xxxx.xxxxx), accepted
  at [AT4SSL 2023](https://sites.google.com/tilburguniversity.edu/at4ssl2023/).

![Visualization of our pipeline](assets/pipeline.jpg)

## Install

```.bash
pip install git+https://github.com/ZurichNLP/spoken-to-signed-translation.git
```

Then, to download a lexicon, run:
```.bash
download_lexicon \
  --name <signsuisse> \
  --directory <path_to_directory>
```

## Usage

For language codes, we use the [IANA Language Subtag Registry](https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry).
Our pipeline provides multiple scripts. 
To quickly demo it using a dummy lexicon, run:

```bash
text_to_gloss_to_pose \
  --text "Kleine Kinder essen Pizza" \
  --glosser "simple" \
  --lexicon "assets/dummy_lexicon" \
  --spoken-language "de" \
  --signed-language "sgg" \
  --pose "quick_test.pose"
```

#### Text-to-Gloss Translation

```bash
# This script translates input text into gloss notation. 
text_to_gloss \
  --text <input_text> \
  --glosser <simple|rules|nmt> \
  --spoken-language <de|fr|it|en> \
  --signed-language <sgg|gsg|bfi>
```

#### Pose-to-Video Conversion

```bash
# This script converts a pose file into a video file.
pose_to_video \
  --pose <pose_file_path>.pose \
  --video <output_video_file_path>.mp4
```

#### Text-to-Gloss-to-Pose Translation

```bash
# This script translates input text into gloss notation, then converts the glosses into a pose file.
text_to_gloss_to_pose \
  --text <input_text> \
  --glosser <simple|rules|nmt> \
  --lexicon <path_to_directory> \
  --spoken-language <de|fr|it|en> \
  --signed-language <sgg|gsg|bfi> \
  --pose <output_pose_file_path>.pose
```

#### Text-to-Gloss-to-Pose-to-Video Translation

```bash
# This script translates input text into gloss notation, converts the glosses into a pose file, and then transforms the pose file into a video.
text_to_gloss_to_pose_to_video \
  --text <input_text> \
  --glosser <simple|rules|nmt> \
  --lexicon <path_to_directory> \
  --spoken-language <de|fr|it|en> \
  --signed-language <sgg|gsg|bfi> \
  --video <output_video_file_path>.mp4
```

#### Example for testing
```bash
text_to_gloss_to_pose_to_video \
  --text "Kleine Kinder essen Pizza" \
  --glosser "rules" \
  --lexicon "assets/dummy_lexicon" \
  --spoken-language "de" \
  --signed-language "ch" \
  --video "example.mp4"
  ```


## Methodology

The pipeline consists of three main components:

1. **Text-to-Gloss Translation:**
   Transforms the input (spoken language) text into a sequence of glosses.

- [Simple lemmatizer](src/text_to_gloss/simple.py),
- [Rule-based word reordering and dropping](src/text_to_gloss/rules.py) component
- [Neural machine translation system](src/text_to_gloss/nmt.py).

2. **Gloss-to-Pose Conversion:**

- [Lookup](src/gloss_to_pose/lookup.py): Uses a lexicon of signed languages to convert the sequence of glosses into a
  sequence of poses.
- [Pose Concatenation](src/gloss_to_pose/concatenate.py): The poses are then cropped, concatenated, and smoothed,
  creating a pose representation for the input sentence.

3. **Pose-to-Video Generation:** Transforms the processed pose video back into a synthesized video using an image
   translation model.

## Supported Languages

| Language                   | Lemmatizers Supported             | Lexicon Data Source                                  |
|----------------------------|-----------------------------------|------------------------------------------------------|
| Swiss German Sign Language | Simplemma, Rule-Based             | [SignSuisse (de)](https://signsuisse.sgb-fss.ch/de/) |
| French Sign Language       | Simplemma                         | [SignSuisse (fr)](https://signsuisse.sgb-fss.ch/fr/) |
| Italian Sign Language      | Simplemma                         | [SignSuisse (it)](https://signsuisse.sgb-fss.ch/it/) |
| German Sign Language       | Simplemma, [NMT](TODO-model-link) | WordNet (Coming Soon)                                |
| British Sign Language      | Simplemma, [NMT](TODO-model-link) | WordNet (Coming Soon)                                |


## Citation

If you find this work useful, please cite our paper:

```.bib
@article{author2022gloss,
    title={Gloss-Based Baseline for Spoken to Signed Language Translation},
    author={Author, Firstname and Coauthor, Firstname},
    journal={arXiv preprint arXiv:xxxx.xxxxx},
    year={2022}
}
```