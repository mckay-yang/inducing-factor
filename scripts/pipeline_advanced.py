"""Legacy placeholder for the experimental full-extent pipeline.

The production-ready pipeline entrypoint is now `inducing_factor.PipelineProcessor`.
Keep this script only for explicit advanced experiments.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inducing_factor.pipeline.processor import PipelineProcessor


if __name__ == "__main__":
    processor = PipelineProcessor(start_year=2000, end_year=2020, chunk_size=1024)
    processor.process_all_chunks()
