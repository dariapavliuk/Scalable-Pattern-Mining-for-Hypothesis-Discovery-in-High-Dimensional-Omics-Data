# Scalable Pattern Mining for Hypothesis Discovery in High-Dimensional Omics Data

This repository contains code for scalable pattern mining and hypothesis discovery in high-dimensional omics data.

## Overview

The project is organized around two main experiment pipelines:

- **SyFlow-based experiments**
- **OmicsARules-based experiments**

It also includes utility scripts for launching experiments and processing results.

## Repository structure

```text
.
├── run_experiments/
│   ├── apply_syflow/
│   │   ├── configs/
│   │   └── run_syflow_pipeline.py
│   └── apply_omicsarules/
│       ├── configs/
│       ├── r_script_omicsarules.R
│       └── run_omicsarules.py
├── src/
│   ├── apply_syflow_omicarules.py
│   └── process_syflow_results.py
└── README.md
