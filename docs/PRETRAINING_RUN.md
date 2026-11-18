# Candidate generation results

The run read 227,577 cleaned observations across 707 segments. Every observation received at least one candidate. Generation checked unique observation–arc pairs and finite input features. A separate read of all 90 persisted feature tables counted 1,763,701 rows.

| Partition | GPS points | Candidate rows | Ambiguous proxies | Unambiguous proxies outside shortlist | Unambiguous observations available for fitting or scoring |
|---|---:|---:|---:|---:|---:|
| Train | 208,417 | 1,618,661 | 32,889 | 16,874 | 158,654 |
| Student-test | 19,160 | 145,040 | 3,924 | 12 | 15,224 |

The training folds contain 15, 16, 16, 17 and 16 trips. All ten student-test trips have fold -1.

## Interpretation

Training trips 00000054 and 00000033 account for 12,168 and 3,832 clear misses respectively. Among missed observations the median distance to the nearest reference arc was 161 m and 133 m respectively. The median distance to the nearest generated candidate was 86 m and 7 m. In particular, trip 33 often has nearby mapped roads while its supplied route geometry is farther away. The evidence leaves both GPS and reference-path error possible. Keep these trips visible in trip-level metrics. An absent proxy cannot be recovered by a classifier operating on the frozen shortlist.

The student-test miss count is descriptive. Candidate settings were fixed before this coverage report. The two partitions have substantially different coverage, so report both the fraction with usable supervision and conditional ranking performance for every future experiment.
