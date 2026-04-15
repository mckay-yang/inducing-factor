#!/usr/bin/env python3
"""
Run pipeline processor but stop after processing N valid chunks.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inducing_factor.pipeline.processor import PipelineProcessor
import rasterio
from rasterio.windows import Window
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FewChunksPipeline")

class FewChunksProcessor(PipelineProcessor):
    def __init__(self, start_year=1990, end_year=2020, chunk_size=1024, cache_dir="/Volumes/T9/Datasets/.cache/chunks", max_chunks=5):
        super().__init__(start_year, end_year, chunk_size, cache_dir)
        self.max_chunks = max_chunks
    
    def process_all_chunks(self):
        logger.info("Starting pipeline processing (max %d valid chunks). Output cache: %s", self.max_chunks, self.cache_dir)
        self.road_loader.load()
        self.river_loader.load()
        with rasterio.open(self.dem_loader.file_path) as src_dem:
            width = src_dem.width
            height = src_dem.height
            transform = src_dem.transform
            chunk_idx = 0
            valid_chunks = 0
            for row_off in range(0, height, self.chunk_size):
                for col_off in range(0, width, self.chunk_size):
                    w = min(self.chunk_size, width - col_off)
                    h = min(self.chunk_size, height - row_off)
                    window = Window(col_off, row_off, w, h)
                    # Check if window likely has data by reading a small sample
                    # We'll just process and see if it returns
                    logger.info("Processing chunk %d at (%d, %d)", chunk_idx, row_off, col_off)
                    self._process_single_chunk(window, transform, chunk_idx, row_off, col_off, src_dem.crs, src_dem.transform, src_dem.width, src_dem.height)
                    # Check if cache file was created
                    import os
                    cache_file = os.path.join(self.cache_dir, f"chunk_{chunk_idx}.npz")
                    if os.path.exists(cache_file):
                        valid_chunks += 1
                        logger.info("Chunk %d produced data (%d valid pixels)", chunk_idx, valid_chunks)
                        if valid_chunks >= self.max_chunks:
                            logger.info("Reached max valid chunks (%d)", self.max_chunks)
                            return
                    chunk_idx += 1
            logger.info("Processed all chunks, found %d valid chunks", valid_chunks)

def main():
    # Use years 2015-2021, smaller chunk size maybe 512 to get data faster
    processor = FewChunksProcessor(start_year=2015, end_year=2021, chunk_size=512, cache_dir="results/cache_real", max_chunks=2)
    processor.process_all_chunks()

if __name__ == "__main__":
    main()