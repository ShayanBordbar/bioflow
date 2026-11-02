# The isolated sequence runtime.
#
# Built from the repository root:
#     docker build -f environments/sequence/Dockerfile -t bioflow/sequence-sidecar .
#
# Note what is NOT copied: src/bioflow. The orchestrator is not present in this image at
# all, so `import bioflow.core` fails with ImportError rather than succeeding quietly.
# That is the isolation invariant made structural -- tests/contract/test_sidecar.py
# checks the same property in-repo, but this is what enforces it in production.
FROM python:3.12-slim

# Fail fast and log unbuffered: a sidecar's stdout is streamed to a file the runner tails
# on failure, and buffered output is empty exactly when it is needed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/sidecar

COPY environments/sequence/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --require-hashes=false -r requirements.txt

COPY src/bioflow_sequence ./bioflow_sequence
ENV PYTHONPATH=/opt/sidecar

# The container runs as an unprivileged user by default; the runner additionally passes
# -u <uid>:<gid> so outputs land owned by the caller rather than by root.
RUN useradd --create-home --uid 1000 sidecar
USER sidecar

# /work is the single writable bind mount. The image itself runs --read-only.
WORKDIR /work

ENTRYPOINT ["python", "-m", "bioflow_sequence"]
