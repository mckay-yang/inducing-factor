#!/usr/bin/env python3
"""
Run pipeline processor but only process first chunk.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inducing_factor.pipeline.processor import PipelineProcessor
import rasterio
from rasterio.windows import Window
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("OneChunkPipeline")

class OneChunkProcessor(PipelineProcessor):
    def process_all_chunks(self):
        logger.info("Starting single-chunk pipeline processing. Output cache: %s", self.cache_dir)
        self.road_loader.load()
        self.river_loader.load()
        with rasterio.open(self.dem_loader.file_path) as src_dem:
            width = src_dem.width
            height = src_dem.height
            transform = src_dem.transform
            chunk_idx = 0
            # Only first chunk
            row_off = 0
            col_off = 0
            w = min(self.chunk_size, width - col_off)
            h = min(self.chunk_size, height - row_off)
            window = Window(col_off, row_off, w, h)
            self._process_single_chunk(window, transform, chunk_idx, row_off, col_off, src_dem.crs, src_dem.transform, src_dem.width, src_dem.height)
            logger.info("Finished processing chunk %d", chunk_idx)

def main():
    # Use same parameters as advanced config
    processor = OneChunkProcessor(start_year=2015, end_year=2021, chunk_size=1024, cache_dir="results/cache_real")
    processor.process_all_chunks()

if __name__ == "__main__":
    main()