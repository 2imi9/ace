LLM Research Agent
==================

This guide explains how to operate the LLM Research Agent workflow alongside the
core emulator stack. It covers prerequisites, configuration, environment
management, security expectations, and step-by-step workflows for both
experimentation and production-grade studies.

Prerequisites
-------------

* An installed copy of the project following the :doc:`installation` guide.
* Python 3.10 or later with the dependencies listed in ``requirements.txt``.
* Access to research datasets compatible with the emulator training pipeline.
* Credentials for external services (for example, artifact storage or model
  registries) that the agent must integrate with.

Environment setup
-----------------

You can provision a reproducible environment with either Docker or Conda.
Both approaches install the base dependencies from ``requirements.txt`` and can
layer on optional research-specific packages.

Docker environment
~~~~~~~~~~~~~~~~~~~~

Create a Docker image that installs the base requirements and any additional CLI
tooling you rely on for research.

.. code-block:: dockerfile

   FROM python:3.10-slim
   WORKDIR /opt/app
   COPY requirements.txt ./
   RUN pip install --no-cache-dir -r requirements.txt \
       && pip install --no-cache-dir -r requirements-dev.txt
   COPY . .
   CMD ["python", "-m", "fastapi.main"]

Conda environment
~~~~~~~~~~~~~~~~~

Use Conda or Mamba to manage an isolated environment. The ``pip`` step reuses
the same ``requirements.txt`` file, which keeps the environment in sync with
production deployments.

.. code-block:: yaml

   name: llm-research-agent
   channels:
     - conda-forge
   dependencies:
     - python=3.10
     - pip
     - pip:
       - -r requirements.txt
       - -r requirements-dev.txt

Environment variables
---------------------

Configure the following environment variables before launching the agent. The
examples use dummy values; replace them with your own secrets or dataset paths.

``ACE_AGENT_DATA_DIR``
   Directory where the agent reads and writes intermediate scientific artifacts.
``ACE_AGENT_MODEL_CACHE``
   Location for cached LLM weights or prompt templates.
``ACE_AGENT_API_KEY``
   API token for remote model providers or vector stores.
``ACE_AGENT_RESULTS_BUCKET``
   Cloud bucket or shared filesystem used to persist experiment outputs.
``ACE_AGENT_LOG_LEVEL``
   Logging verbosity (``INFO`` by default).

Security best practices
-----------------------

* Store long-lived keys (such as ``ACE_AGENT_API_KEY``) in a secret manager or
  a secure OS keyring rather than hard-coding them in notebooks.
* Limit permissions on data directories (``ACE_AGENT_DATA_DIR`` and
  ``ACE_AGENT_RESULTS_BUCKET``) to the minimal read/write scope required for the
  experiment.
* Rotate keys regularly and audit access logs to detect anomalies.
* Avoid uploading proprietary or export-controlled scientific data to third
  parties unless they are part of an approved collaboration.
* Use encrypted channels (HTTPS, TLS-encrypted storage) whenever data leaves the
  local research environment.

Sample workflows
----------------

The following workflows highlight common research tasks:

* **Exploratory prompt design:** Load a small dataset, iterate on prompts with a
  mocked LLM (see :doc:`examples/llm_research_agent_quickstart`), and evaluate
  outputs offline.
* **Batch experimentation:** Use the Docker or Conda environment above,
  schedule batch runs with environment variables configured via your orchestration
  system, and persist results to ``ACE_AGENT_RESULTS_BUCKET``.
* **Model validation:** Compare mocked or fine-tuned LLM outputs to baseline
  emulator runs and write detailed evaluation reports for peer review.

For an end-to-end example, open the quickstart notebook in
``docs/examples/llm_research_agent_quickstart.ipynb``.
