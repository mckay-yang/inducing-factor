//! Chunked processing pipeline: tiled raster → feature engineering
//! (rainfall windows, rolling history) → stratified background sampling
//! → probability map generation.

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(2 + 2, 4);
    }
}
