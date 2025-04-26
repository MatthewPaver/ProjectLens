# Project Data Directory

This directory contains the data files used by ProjectLens

## Directory Structure

- `input/`: Input project files to be processed
- `output/`: Processed results output by the pipeline
- `archive/`: 
  - `success/`: Successfully processed projects
  - `failed/`: Projects that failed processing

## Data Flow

1. Raw project files are placed in the `input/` directory
2. The pipeline processes files from `input/`
3. Results are written to the `output/` directory
4. Processed input files are moved to `archive/success` or `archive/failed` based on processing outcome
