---
title: "SEDG: A Synthetic Experimental Data Generator for LLM-Based Behavioral Experiments"
tags:
  - Python
  - large language models
  - synthetic data
  - behavioral experiments
  - marketing research
  - consumer behavior
  - survey research
authors:
  - name: Siva Shanmugam Mariappan
    orcid: 0000-0001-8200-3579
    affiliation: 1
    corresponding: true
  - name: Ashwin Malshe
    orcid: 0000-0002-3429-4268
    affiliation: 1
affiliations:
  - name: The University of Texas at San Antonio, United States
    index: 1
date: 1 July 2026
bibliography: paper.bib
---

# Summary

The rising cost, time burden, and ethical constraints associated with collecting human behavioral data create persistent challenges for researchers who design, pilot, and test behavioral theories. These challenges are especially important in marketing, psychology, information systems, management, and social sciences, where experiments often require large samples, multiple conditions, repeated pretests, or sensitive manipulations. Large language models (LLMs) offer a promising complementary approach by generating synthetic participant responses that can help researchers evaluate stimuli, refine survey instruments, explore theoretical predictions, and compare expected response patterns before costly human-subject data collection. However, using LLMs for this purpose often requires substantial technical expertise, including model selection, API configuration, prompt engineering, persona construction, response parsing, and reproducible data management.

`SEDG` (Synthetic Experimental Data Generator) is a software platform that lowers these barriers by enabling researchers to generate synthetic responses for behavioral experiments using local and cloud-based LLMs. The platform is available at <https://synthstudy.vercel.app/> and is designed around the workflow of experimental researchers rather than software engineers. `SEDG` allows users to specify study stimuli, define survey questions, select LLM providers and models, configure generation parameters, use predefined personas, provide custom personas, or generate responses without personas. It supports diverse experimental designs and produces structured outputs for analysis, replication, and comparison across models and conditions. By integrating flexible LLM access with research-oriented prompt construction and export functionality, `SEDG` democratizes access to LLM-based behavioral simulation across disciplines.

# Statement of need

Behavioral researchers increasingly face practical and methodological constraints when collecting human-subject data. Online panels and crowdsourcing platforms have expanded access to participants, but they also raise concerns about inattentive respondents, bots, duplicate participation, professional survey takers, platform-specific sampling biases, and rising participant costs. For early-stage research, these challenges make it difficult to test whether a manipulation is understandable, whether a survey item is clear, or whether a theoretically predicted pattern is plausible before launching a full study. For researchers with limited funding, the cost of repeated pretesting can constrain the number of theoretical ideas that can be explored.

At the same time, LLMs are increasingly being considered as synthetic respondents, simulated consumers, or digital twins that may support survey research and behavioral experimentation when used with appropriate validation [@huang_rust_2025; @ghasemi_2026; @toubia_2025]. These models can generate structured responses from the perspective of specified personas, react to experimental stimuli, and produce data that researchers can inspect before collecting human responses. Such capabilities are not a replacement for human-subject research. Instead, they can serve as a complement for piloting materials, stress-testing designs, comparing model behavior across conditions, and identifying cases where synthetic responses diverge from established theory or empirical evidence. LLMs can also serve as objects of study in their own right, enabling researchers to examine how different models respond to the same experimental stimuli.

Despite this promise, the practical use of LLM-based synthetic respondents remains limited. Proprietary LLMs provide convenient access to frontier models but raise concerns about cost, privacy, reproducibility, and dependence on commercial platforms. Open-weight LLMs offer a more transparent and potentially lower-cost alternative, but they often require expertise in local model deployment, hardware constraints, inference settings, and model-specific prompting. Researchers who do not regularly write code may find it difficult to translate a behavioral experiment into a structured LLM workflow. Even technically skilled researchers often need to write custom scripts for each study, which reduces reproducibility and makes it harder to compare outputs across experimental designs.

`SEDG` addresses this gap by providing an accessible platform for designing and running synthetic behavioral experiments with LLMs. Rather than requiring users to build a custom API wrapper or local inference pipeline, `SEDG` offers a structured interface that connects experimental design decisions to model generation settings. This allows researchers to focus on substantive questions: What stimuli should participants see? What conditions should be compared? What survey questions should be asked? Should responses be generated from generic respondents, predefined digital-twin personas, or user-provided personas? Which local or cloud-based model should be evaluated? By making these choices explicit and configurable, `SEDG` supports transparent, reproducible, and comparative use of LLMs in behavioral research.

# Software design

`SEDG` is designed to support the full workflow of synthetic behavioral data generation. Users enter study materials, including instructions, stimuli, manipulations, and survey questions. The software then helps construct prompts that present the experimental task to the selected LLM in a consistent and editable format. Users may rely on the default prompt structure or modify the instructions to match a specific study, model, or research context.

The platform supports both cloud-based and local LLM workflows. Cloud-based models provide fast access to frontier proprietary systems through API keys, while local open-weight models offer a privacy-preserving alternative for researchers who prefer not to send study materials to external providers. This dual design is important because behavioral researchers often work with unpublished theories, novel stimuli, proprietary materials, or sensitive research ideas. By supporting both deployment modes, `SEDG` allows users to balance speed, cost, privacy, and reproducibility.

A central feature of `SEDG` is its flexible persona system. Users can generate synthetic responses without personas, use predefined digital-twin-style personas, or provide custom persona descriptions. Persona-based generation is useful when researchers want to examine whether response patterns differ across demographic, psychological, or behavioral profiles. The option to run studies without personas allows users to evaluate model behavior under a more generic respondent framing. This flexibility helps researchers compare whether personas meaningfully affect response distributions, treatment effects, and theoretical conclusions.

`SEDG` also supports multiple response formats commonly used in behavioral research. Survey questions may include discrete scale responses, continuous numeric responses, or text-based answers. The software produces structured output that can be downloaded and analyzed using common statistical tools. This output-oriented design is critical because synthetic data generation is most actionable when researchers can inspect, clean, analyze, and replicate the generated responses. By producing research-friendly files, `SEDG` reduces the need for ad hoc parsing and makes it easier to compare outputs across models, prompts, personas, and experimental conditions.

# Research impact statement

`SEDG` is intended for researchers who want to incorporate LLM-based synthetic response generation into the early stages of behavioral research. In marketing and consumer research, the platform can be used to pilot advertising stimuli, product descriptions, consumer risk scenarios, pricing manipulations, brand communications, service failures, or public policy messages. In psychology and the social sciences, it can support early tests of survey wording, vignette clarity, theory-driven manipulations, and response heterogeneity across personas. In information systems and human-computer interaction, it can help researchers examine responses to AI interfaces, privacy notices, recommendation systems, and technology adoption scenarios.

`SEDG` provides a systematic environment for using LLMs as a complementary research tool. When used for pretesting and theory development, it can help researchers identify promising manipulations, detect ambiguous materials, compare model sensitivity, and generate preliminary predictions before investing in full-scale data collection.

By reducing technical barriers, `SEDG` broadens access to LLM-based behavioral simulation. Researchers who lack programming expertise can evaluate multiple models and experimental conditions, while technically advanced users can benefit from a standardized and reproducible workflow. This makes the platform useful for research teams, doctoral seminars, methods courses, and interdisciplinary projects where participants vary in computational background.

# AI usage disclosure

Large language models were used to assist with portions of software development. The authors reviewed, modified, and validated AI-generated content and retained responsibility for the software architecture, research design decisions, and implementation choices.

# Acknowledgements

The authors thank colleagues and research participants whose feedback informed the development of `SEDG`. The authors also acknowledge ongoing discussions in marketing, psychology, and computational social science about the appropriate use, validation, and limitations of LLM-generated synthetic behavioral data.

# References
