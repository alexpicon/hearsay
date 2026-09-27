# HEARSAY test audio

Author: Alex Picon <alexnpc@me.com>

Four labeled clips selected without examining detector scores, from the
ASVspoof 2019 Logical Access evaluation partition. Files are byte-for-byte copies
of the local benchmark FLAC inputs, renamed for convenience. No audio was changed.
Labels and original IDs/hashes are in labels.json; 0 = real, 1 = synthetic/spoof.

- real-1.flac: real human speech, 3.37 seconds. Original ID: la19/test/LA_E_9102181; attack: bonafide.
- real-2.flac: real human speech, 3.41 seconds. Original ID: la19/test/LA_E_4936594; attack: bonafide.
- synthetic-1.flac: synthetic/spoofed speech, 2.94 seconds. Original ID: la19/test/LA_E_8970871; attack: A17.
- synthetic-2.flac: synthetic/spoofed speech, 2.29 seconds. Original ID: la19/test/LA_E_9760280; attack: A19.

## Source and attribution

Audio: ASVspoof 2019 database, provided by the ASVspoof organizing committee:
Junichi Yamagishi, Massimiliano Todisco, Md Sahidullah, Héctor Delgado, Xin Wang,
Nicholas Evans, Tomi Kinnunen, Kong Aik Lee, Ville Vestman and Andreas Nautsch.
Alex Picon assembled this sample pack; the source audio belongs to its credited
contributors.

Source: https://zenodo.org/records/6906306
DOI: https://doi.org/10.7488/ds/2555
License: Open Data Commons Attribution License (ODC-By) 1.0,
https://opendatacommons.org/licenses/by/1-0/
This sample pack contains information from ASVspoof 2019, which is made available
here under the Open Data Commons Attribution License (ODC-By) v1.0.

## Use

Extract the ZIP, enter your dashboard access code, then upload one FLAC file at a
time and run the selected detectors. A higher raw score means more synthetic for
each detector; numeric scales differ between detectors.

These are demonstration files, not an independent accuracy test. Our model was
trained using material from this partition; other detectors may also have seen
related data. Labels describe dataset ground truth, not guaranteed detector
answers. Never rank the systems from four examples.
