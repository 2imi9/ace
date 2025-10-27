[![Docs](https://readthedocs.org/projects/ai2-climate-emulator/badge/?version=latest)](https://ai2-climate-emulator.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/fme.svg)](https://pypi.org/project/fme/)

<img src="ACE-logo.png" alt="Logo for the ACE Project" style="width: auto; height: 50px;">

# Ai2 Climate Emulator

Ai2 Climate Emulator (ACE) is a fast machine learning model that simulates global atmospheric variability in a changing climate over time scales ranging from hours to centuries.

This repo contains code accompanying five papers describing ACE models:
- "ACE: A fast, skillful learned global atmospheric model for climate prediction" ([link](https://arxiv.org/abs/2310.02074))
- "Application of the Ai2 Climate Emulator to E3SMv2's global atmosphere model, with a focus on precipitation fidelity" ([link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JH000136))
- "ACE2: Accurately learning subseasonal to decadal atmospheric variability and forced responses" ([link](https://www.nature.com/articles/s41612-025-01090-0))
- "ACE2-SOM: Coupling an ML Atmospheric Emulator to a Slab Ocean and Learning the Sensitivity of Climate to Changed CO<sub>2</sub>" ([link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024JH000575))
- "Applying the ACE2 Emulator to SST Green's Functions for the E3SMv3 Global Atmosphere Model" ([link](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000774))

## Installation

```
pip install fme
```

## Documentation

See complete documentation [here](https://ai2-climate-emulator.readthedocs.io/en/latest/) and a quickstart guide [here](https://ai2-climate-emulator.readthedocs.io/en/latest/quickstart.html).

### Building an ACE-powered research assistant

If you would like to create an ACE-based agent that can orchestrate climate
simulations while collaborating with hosted large language models (for example
OpenAI, Anthropic Claude, or Google Gemini), we recommend the following
high-level roadmap. A short "How to use it" guide follows immediately after the
roadmap so you can see what day-to-day operation of the agent could look like
once the pieces are assembled.

1. **Establish an agent skeleton** – Create a new `fme/agent/` package that can
   coordinate ACE steppers and aggregators. Expose commands for running
   simulations, analyzing outputs, and dispatching LLM requests. Support
   configuration through environment variables or YAML files so different
   deployments can choose providers and defaults.
2. **Implement multi-provider LLM clients** – Add provider wrappers under
   `fme/agent/providers/` implementing a common `LLMClient` interface with
   retry/backoff logic. Load credentials from the environment and register the
   providers in a factory for easy selection. Consider automating provider or
   model selection by letting the orchestrator inspect the incoming request,
   run quick ACE diagnostics if needed, and then choose the most suitable LLM
   configuration (e.g., higher-context models for complex simulations, cheaper
   ones for routine status updates).
3. **Hook ACE simulations into the agent** – Reuse existing utilities in
   `fme/ace/` to prepare inputs, launch simulations, and collect diagnostics.
   Store structured outputs (e.g., NetCDF, Parquet) and expose metadata that can
   be summarized for LLM prompts.
4. **Create scientific prompt templates** – Under `fme/agent/prompts/`, define
   templates that describe datasets, assumptions, and requested analyses. Ensure
   the prompt builder injects variable descriptions and uncertainty information
   before calling the LLM.
5. **Add validation and safety mechanisms** – Implement verification routines in
   `fme/agent/validation.py` so LLM claims are compared against ACE results or
   known constraints. Add guardrails (content filters, citation requirements,
   refusal handling) and log every interaction for reproducibility.
6. **Ship user-facing interfaces** – Provide a CLI (and optionally a REST API)
   so researchers can submit questions, monitor runs, and retrieve analyses.
   Include integration tests that exercise these workflows with mocked LLM
   providers.
7. **Document setup and deployment** – Extend the documentation with setup
   guides, environment configuration, and security recommendations. Consider
   adding an example notebook that demonstrates an end-to-end analysis with a
   mocked provider.

This roadmap leverages the existing simulation infrastructure in `fme/ace/`
while keeping LLM integrations modular and secure.

#### How to use the agent once it is wired together

The blueprint above deliberately separates design and execution. When you are
ready to operate the agent, the typical workflow looks like this:

1. **Install runtime dependencies** – Create a virtual environment and install
   ACE alongside any LLM SDKs you plan to talk to:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   pip install openai anthropic google-generativeai
   ```

2. **Provide credentials and defaults** – Export the API keys that correspond to
   the provider classes you implemented and point the agent at its experiment
   workspace:

   ```bash
   export OPENAI_API_KEY=...
   export ANTHROPIC_API_KEY=...
   export GOOGLE_API_KEY=...
   export ACE_AGENT_HOME=$PWD/.ace-agent
   ```

   You can also create a YAML configuration file (for example,
   `configs/agent-defaults.yaml`) that mirrors the structure consumed by your
   orchestrator:

   ```yaml
   provider: openai
   model: gpt-4.1
   simulation_defaults:
     dataset: era5
     lead_days: 30
   caching:
     enabled: true
     max_age_hours: 12
   ```

3. **Launch simulations or chat sessions** – With credentials and configuration
   in place, you can interact with the agent through the CLI or programmatic
   APIs:

   ```bash
   # Submit a research question from the command line
   python -m fme.agent.cli ask "How will ENSO anomalies influence North American rainfall in the next 90 days?"

   # Trigger a scripted workflow that runs a simulation bundle
   python scripts/run_ace_agent.py configs/agent-defaults.yaml
   ```

   From Python code you might drive the orchestrator directly:

   ```python
   from fme.agent.orchestrator import ResearchAgent

   agent = ResearchAgent.from_config("configs/agent-defaults.yaml")
   report = agent.answer_question(
       "Summarize projected ENSO-driven precipitation anomalies",
       region="North America",
       lead_time_days=90,
   )
   print(report.summary)
   ```

4. **Review artefacts** – Each interaction should log prompts, model choices,
   and ACE outputs under `ACE_AGENT_HOME`. This makes it easy to revisit the
   diagnostics the agent ran, audit the LLM responses, or attach artefacts to a
   lab notebook.

5. **Iterate safely** – Before promoting new providers or prompt templates,
   execute your automated tests (including guardrail checks) so that scientific
   claims remain traceable to ACE data products.

This operational flow is intentionally provider-agnostic: switching from OpenAI
to Gemini (or any future backend) simply requires swapping the configuration and
ensuring the corresponding API key is available in the environment.

## Model checkpoints

Pretrained model checkpoints are available in the [ACE Hugging Face](https://huggingface.co/collections/allenai/ace-67327d822f0f0d8e0e5e6ca4) collection.

## Available datasets
Two versions of the complete dataset described in [arxiv:2310.02074](https://arxiv.org/abs/2310.02074)
are available on a [requester pays](https://cloud.google.com/storage/docs/requester-pays) Google Cloud Storage bucket:
```
gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-zarrs
gs://ai2cm-public-requester-pays/2023-11-29-ai2-climate-emulator-v1/data/repeating-climSST-1deg-netCDFs
```
The `zarr` format is convenient for ad-hoc analysis. The netCDF version contains our
train/validation split which was used for training and inference.

The datasets used in the [ACE2 paper](https://arxiv.org/abs/2411.11268) are available at:
```
gs://ai2cm-public-requester-pays/2024-11-13-ai2-climate-emulator-v2-amip/data/c96-1deg-shield/
gs://ai2cm-public-requester-pays/2024-11-13-ai2-climate-emulator-v2-amip/data/era5-1deg-1940-2022.zarr/
```

The dataset used in the [ACE2-SOM paper](https://arxiv.org/abs/2412.04418) is available at:
```
gs://ai2cm-public-requester-pays/2024-12-05-ai2-climate-emulator-v2-som/SHiELD-SOM-C96
```
