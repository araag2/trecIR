# TREC 2021 - Query Relevance

- **Not Relevant (0 in raw)** The patient is not relevant for the trial in any way.

- **Excluded (1 in raw)** The patient has the condition that the trial is targeting, but the exclusion criteria make
  the patient ineligible.

- **Eligible (2 in raw)** The patient is eligible to enroll in the trial.

## qrels<year>.json

| Not Relevant | Excluded | Eligible |
|:------------:|:--------:|:--------:|
|      0       |    1     |    2     |

## qrels<year>_binary.json

| Not Relevant | Excluded | Eligible |
|:------------:|:--------:|:--------:|
|      0       |    0     |    1     |

## qrels<year>_similiar.json

| Not Relevant | Excluded | Eligible |
|:------------:|:--------:|:--------:|
|      0       |    1     |    1     |