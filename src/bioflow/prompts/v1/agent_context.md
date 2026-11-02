## Current state: {{ state }}

{{ objective }}

## The request

```json
{{ request }}
```

## Tools available to you here

{{ tools }}

Call exactly one of them. No other tool exists in this state.

## Artifacts in this run

Each row is `artifact_id | type | summary | size`. You refer to an artifact by its
identifier and nothing else — there are no filenames or paths, and asking for one is not
a thing this system can answer.

{{ artifacts }}

## What has happened so far

{{ history }}

Anything marked `[truncated: ...]` was cut to fit a budget. Where you see one, you are
looking at part of something, and you should say so rather than describing the part as
though it were the whole.
