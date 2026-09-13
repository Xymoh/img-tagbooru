# Img-Tagbooru — Terms of Use, Acceptable Use and Privacy Notice

**Last updated:** 13 September 2026 · **Applies to:** Img-Tagbooru v1.3.7 and later

Img-Tagbooru ("the Software") is free, open-source software published by Szymon Ruszkiewicz ("the Author") under the MIT License (see `LICENSE`). These Terms supplement the MIT License. By downloading, installing, running or building the Software you agree to them. If you do not agree, do not use the Software.

The Software is a local desktop tool that produces **text** (Danbooru-style tags, captions and prompts) from images or text you supply, using machine-learning models that you download and run on your own computer. It does not generate images, audio or video, and it does not host, upload or share anything.

---

## 1. Age requirement

The Software can produce adult and sexually explicit **text** when you enable the Mature mode, the Explicit option, or use a model that permits such output. You must be **at least 18 years old**, or the age of majority where you live if that is higher, to use the Software. The Software asks you to confirm this on first launch.

## 2. Acceptable use

You are solely responsible for what you do with the Software and its output. You agree **not** to use the Software:

1. **In connection with any sexual depiction of a minor.** This includes real, drawn, animated, rendered, AI-generated or fictional persons who are, appear to be, or are presented as under 18. It includes producing tags, captions or prompts for such material and using the Software to caption or prepare training data for it. This is a criminal offence in the European Union, the United States, the United Kingdom, Poland and most other jurisdictions, and it is the one category of content the Software actively refuses to tag: age-descriptor tags are removed from every output path in the Software's own code (`backend/content_policy.py`).
2. **To create or facilitate non-consensual intimate content of real, identifiable people**, including "deepfake" or synthetic sexual imagery of anyone who has not consented, or content that harasses, threatens or defames a real person.
3. In any way that violates the law that applies to you, the rights of any third party, or the terms of any third-party model, dataset or service you use with the Software (see section 4).
4. To train, fine-tune or caption datasets whose use would infringe someone else's copyright, personality rights or other rights.

The Author may refuse support, close issues, or remove community access for anyone who breaches this section.

## 3. Adult content

The Software does not filter lawful adult content. Whether such content is produced depends on the modes and options you choose and on the models you decide to install. Producing, possessing or distributing adult content is regulated differently in different countries. It is your responsibility to know and comply with the law where you are.

## 4. Third-party models, data and services

The Software does not ship any machine-learning model. It downloads or connects to models you choose:

- **WD tagger models** by SmilingWolf are downloaded from Hugging Face at your request (Apache License 2.0).
- **Language and vision models** are pulled and run by you through **Ollama**. Each model has its own licence and usage policy. Some models the documentation mentions are derived from Llama 3.1 (Meta Llama 3.1 Community License and Acceptable Use Policy), Qwen (Apache 2.0), or Mistral (Apache 2.0). It is your responsibility to read and comply with the licence of every model you install. The Author is not a party to those licences.
- The Software bundles a **Danbooru tag vocabulary** (tag names and post counts). Danbooru's Terms of Service state that tags are factual information and not copyrightable. No images from Danbooru or any other site are included.
- The Software optionally fetches an image from a **URL you paste**, checks GitHub for **updates** when you run the update script, and talks to **Ollama on localhost**. No other network access occurs.

## 5. Privacy

The Software collects **no** personal data and contains **no** telemetry, analytics, crash reporting or account system. Images, captions, prompts and settings stay on your computer. The only data that leaves your machine is:

- an ordinary HTTPS request to **huggingface.co** when you download a recognition model,
- an ordinary HTTPS request to **api.github.com** when you run the update script,
- a request to whatever **URL you yourself paste** as an image source.

Those services see your IP address and process it under their own privacy policies. The Author receives nothing. Because the Author processes no personal data, the Author is not a data controller under the GDPR in respect of your use of the Software. Settings and model files are stored under `~/.img_tagger` on your computer and you may delete them at any time.

## 6. No warranty, limitation of liability

The Software is provided **"as is"**, without warranty of any kind, express or implied, as set out in the MIT License. Model output is statistical and may be wrong, offensive or unsuitable; you must review it before relying on it. To the maximum extent permitted by applicable law, the Author is not liable for any claim, damages or other liability arising from the Software or your use of it, including your use of any third-party model or service. Nothing in these Terms limits liability that cannot be limited under the law that applies to you (for example, liability for intentional misconduct or, for consumers in the EU, statutory rights that cannot be waived).

## 7. Open source, contributions and donations

The source code is licensed under the MIT License and may be used, modified and redistributed under its conditions, including keeping the copyright and licence notice. Third-party components bundled in the executable are listed, with their licences, in `THIRD_PARTY_NOTICES.txt`. Donations through Ko-fi are voluntary gifts that support development; they do not purchase a product, a service, support, or any feature, and they do not create a consumer sale.

## 8. Changes and contact

These Terms may be updated when the Software changes. The version and date at the top identify the current Terms, and the Software will ask you to accept a changed version on the next launch. Questions and reports of misuse can be raised at <https://github.com/Xymoh/img-tagboru-ai/issues>.

## 9. Governing law

These Terms are governed by the law of the Republic of Poland, without prejudice to any mandatory consumer-protection rules of the country where you live. Any dispute is subject to the courts of Poland unless mandatory law provides otherwise.

---

*This document is provided in good faith by an independent developer and is not legal advice.*
