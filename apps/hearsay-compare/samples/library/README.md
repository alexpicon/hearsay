# Recording library

Author: Alex Picon <alexnpc@me.com>

122 MP3 demonstration clips restored byte-for-byte from the original recording
library: 96 public-figure clips and 26 modern speech clips. The public-figure
selection includes four real/synthetic JFK pairs. Catalog labels come from the
source datasets; they are not predictions from the current HEARSAY models.
No historical detector scores are included. Use the website to run fresh scores.

`catalog.json` records each clip's source URL, supplied label, duration, original
identifier and SHA-256. These are demonstration samples, not an independent
accuracy benchmark; training overlap is possible. Synthetic clips imitate the
named speaker and should not be presented as authentic recordings.

## Source credits

Public-figure clips: In-the-Wild audio deepfake dataset, Müller et al.,
Interspeech 2022, distributed through SpeechAntiSpoofingBenchmarks:
https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/InTheWild
The original library identifies the dataset packaging as Apache-2.0; underlying
recordings remain attributed to their source speakers and rights holders.

Modern clips come from the dataset sources linked individually in the catalog,
including Gary Stafford's deepfake-audio-detection collection, WpythonW's
ElevenLabs technical-speech collection, LibriSpeech and FLEURS. Generator names
identify the supplied sample provenance, not endorsements or HEARSAY attribution
predictions. Consult each source's terms before reusing the recordings elsewhere.
Alex Picon assembled the interface; authorship of the source recordings is not
claimed.
