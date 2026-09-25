# Term coverage in retrieved governance sources

Raw case-insensitive match counts in the extracted text saved under `phase1/sources/`. A zero means the term does not occur in the retrieved text. A nonzero count does not show adequate treatment. Extraction details and hashes are in `phase1/sources/manifest.json`.

Characters: S1 NIST AI RMF 1.0 106,437; S2 EU AI Act Art. 14 + 72 20,214; S3 FDA PCCP guidance 157,490

| Term (regex) | S1 NIST AI RMF 1.0 (full text) | S2 EU AI Act Art. 14 + 72 (mirror; 2 articles only) | S3 FDA PCCP guidance (full text) |
|---|---|---|---|
| monitor* (`monitor`) | 19 | 24 | 13 |
| calibration (excluding recalibration) (`(?<![a-z])calibrat`) | 1 | 0 | 0 |
| recalibration (an action) (`recalibrat`) | 3 | 0 | 0 |
| uncertainty (`uncertaint`) | 6 | 0 | 1 |
| drift (`drift`) | 1 | 0 | 2 |
| dataset/distribution shift (`(dataset\|distribution\|data)\s+shift`) | 0 | 0 | 1 |
| automation bias (`automation\s+bias\|over-relying`) | 0 | 2 | 0 |
| override (`overrid`) | 1 | 1 | 0 |
| near miss (`near[\s-]miss`) | 0 | 0 | 0 |
| threshold (`threshold`) | 1 | 0 | 0 |
| detection time (`time[\s-]to[\s-]detect\|detection\s+time\|time\s+to\s+identif`) | 0 | 0 | 0 |
