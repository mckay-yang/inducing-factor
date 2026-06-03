use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rayon::prelude::*;

use crate::processor::{ChunkResult, ChunkTask};

pub fn execute_chunk_tasks<F>(
    tasks: &[ChunkTask],
    worker_count: usize,
    process_fn: F,
) -> Vec<ChunkResult>
where
    F: Fn(&ChunkTask, usize) -> ChunkResult + Sync + Send,
{
    if tasks.is_empty() {
        return vec![];
    }

    if worker_count <= 1 {
        return tasks.iter().map(|task| process_fn(task, 0)).collect();
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .expect("Failed to build rayon thread pool");

    let total = Arc::new(AtomicUsize::new(0));
    let task_count = tasks.len();

    let results: Vec<ChunkResult> = pool.install(|| {
        tasks
            .par_iter()
            .enumerate()
            .map(|(worker_idx, task)| {
                let result = process_fn(task, worker_idx);
                let completed = total.fetch_add(1, Ordering::Relaxed) + 1;
                if completed.is_multiple_of(10) || completed == task_count {
                    log::info!("Chunk progress: {}/{} completed", completed, task_count);
                }
                result
            })
            .collect()
    });

    results
}

pub fn execute_chunk_tasks_bounded<F>(
    tasks: &[ChunkTask],
    max_workers: usize,
    memory_limit_gb: Option<f64>,
    estimated_worker_gb: f64,
    process_fn: F,
) -> Vec<ChunkResult>
where
    F: Fn(&ChunkTask, usize) -> ChunkResult + Sync + Send,
{
    let cpu_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let cpu_limit = cpu_workers.min(max_workers);

    let memory_limit = memory_limit_gb
        .filter(|_| estimated_worker_gb > 0.0)
        .map(|mem_gb| (mem_gb / estimated_worker_gb) as usize)
        .unwrap_or(usize::MAX);

    let worker_count = cpu_limit.min(memory_limit).max(1);

    execute_chunk_tasks(tasks, worker_count, process_fn)
}

// ── Memory guardrail ─────────────────────────────────────────────

/// Estimate per-worker memory consumption in GiB based on chunk dimensions.
///
/// Formula: chunk pixels × features (~30) × 4 bytes/f32 × 2 (working-copy factor).
/// Returns a conservative upper bound.
pub fn estimate_worker_memory_gb(chunk_width: usize, chunk_height: usize) -> f64 {
    const ESTIMATED_FEATURE_COUNT: f64 = 30.0;
    const BYTES_PER_F32: f64 = 4.0;
    const WORKING_COPY_FACTOR: f64 = 2.5;
    const BYTES_PER_GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    let pixels = (chunk_width * chunk_height) as f64;
    let bytes = pixels * ESTIMATED_FEATURE_COUNT * BYTES_PER_F32 * WORKING_COPY_FACTOR;
    bytes / BYTES_PER_GIB
}

/// Detect total system RAM in GiB. Returns `None` if detection fails.
pub fn detect_system_ram_gb() -> Option<f64> {
    use sysinfo::System;
    let sys = System::new_all();
    let total_kb = sys.total_memory();
    if total_kb == 0 {
        None
    } else {
        Some(total_kb as f64 / (1024.0 * 1024.0))
    }
}

/// Detect available system RAM in GiB. Returns `None` if detection fails.
pub fn detect_available_ram_gb() -> Option<f64> {
    use sysinfo::System;
    let sys = System::new_all();
    let avail_kb = sys.available_memory();
    if avail_kb == 0 {
        None
    } else {
        Some(avail_kb as f64 / (1024.0 * 1024.0))
    }
}

/// Resolve the effective worker count given a raw request and memory constraints.
///
/// Returns the recommended number of workers and an optional warning message if
/// the count was reduced due to memory limits.
pub fn resolve_workers_with_memory_guard(
    raw_workers: usize,
    max_workers: usize,
    estimated_worker_gb: f64,
    memory_limit_gb: Option<f64>,
    memory_fraction: f64,
) -> (usize, Option<String>) {
    let cpu_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let cpu_capped = raw_workers.min(cpu_count).min(max_workers);

    let mem_cap = if estimated_worker_gb <= 0.0 {
        usize::MAX
    } else if let Some(limit) = memory_limit_gb {
        (limit / estimated_worker_gb).floor() as usize
    } else if let Some(avail) = detect_available_ram_gb() {
        let budget = avail * memory_fraction;
        (budget / estimated_worker_gb).floor() as usize
    } else {
        usize::MAX
    };

    let effective = cpu_capped.min(mem_cap).max(1);
    let warning = if effective < cpu_capped && mem_cap != usize::MAX {
        let reason = if memory_limit_gb.is_some() {
            format!(
                "memory_limit_gb constraint (estimated: {:.1} GB/worker)",
                estimated_worker_gb
            )
        } else {
            let avail = detect_available_ram_gb().unwrap_or(0.0);
            format!(
                "available system RAM ({:.1} GB) × memory_fraction ({})",
                avail, memory_fraction
            )
        };
        Some(format!(
            "Reducing workers from {} to {} due to memory constraints — {}",
            cpu_capped, effective, reason
        ))
    } else {
        None
    };

    (effective, warning)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::ChunkAudit;
    use std::sync::Mutex;

    #[test]
    fn test_parallel_chunk_execution() {
        let tasks: Vec<ChunkTask> = (0..8)
            .map(|i| ChunkTask {
                chunk_idx: i,
                row_off: 0,
                col_off: i * 1024,
                width: 1024,
                height: 1024,
            })
            .collect();

        let processed = Arc::new(Mutex::new(Vec::new()));

        let results = execute_chunk_tasks(&tasks, 4, |task, worker_id| {
            processed.lock().unwrap().push((task.chunk_idx, worker_id));
            ChunkResult {
                chunk_idx: task.chunk_idx,
                saved: true,
                cache_file: None,
                audit: ChunkAudit::empty(task.chunk_idx, "complete"),
            }
        });

        let p = processed.lock().unwrap();
        let unique_workers: std::collections::HashSet<usize> = p.iter().map(|(_, w)| *w).collect();
        assert!(
            unique_workers.len() > 1,
            "Expected >1 unique worker IDs, got {}",
            unique_workers.len()
        );

        assert_eq!(results.len(), 8);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.chunk_idx, i);
            assert!(r.saved);
        }
    }

    #[test]
    fn test_parallel_execution_empty_tasks() {
        let tasks: Vec<ChunkTask> = vec![];
        let results = execute_chunk_tasks(&tasks, 4, |_, _| ChunkResult {
            chunk_idx: 0,
            saved: false,
            cache_file: None,
            audit: ChunkAudit::empty(0, "skip"),
        });
        assert!(results.is_empty());
    }

    #[test]
    fn test_parallel_execution_single_task() {
        let tasks = vec![ChunkTask {
            chunk_idx: 0,
            row_off: 0,
            col_off: 0,
            width: 512,
            height: 512,
        }];
        let results = execute_chunk_tasks(&tasks, 2, |task, _wid| ChunkResult {
            chunk_idx: task.chunk_idx,
            saved: true,
            cache_file: None,
            audit: ChunkAudit::empty(task.chunk_idx, "complete"),
        });
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_idx, 0);
        assert!(results[0].saved);
    }

    #[test]
    fn test_worker_bounded_by_memory() {
        let tasks: Vec<ChunkTask> = (0..4)
            .map(|i| ChunkTask {
                chunk_idx: i,
                row_off: 0,
                col_off: i * 256,
                width: 256,
                height: 256,
            })
            .collect();

        let results =
            execute_chunk_tasks_bounded(&tasks, 8, Some(2.0), 1.5, |task, _| ChunkResult {
                chunk_idx: task.chunk_idx,
                saved: true,
                cache_file: None,
                audit: ChunkAudit::empty(task.chunk_idx, "complete"),
            });

        assert_eq!(results.len(), 4);
        assert!(results.iter().all(|r| r.saved));
    }

    #[test]
    fn test_skip_resume_audit() {
        let tasks: Vec<ChunkTask> = (0..10)
            .map(|i| ChunkTask {
                chunk_idx: i,
                row_off: 0,
                col_off: i * 1024,
                width: 1024,
                height: 1024,
            })
            .collect();

        let skip_set: std::collections::HashSet<usize> = [0, 2, 4, 6, 8].iter().copied().collect();

        let results: Vec<ChunkResult> = tasks
            .iter()
            .map(|task| {
                if skip_set.contains(&task.chunk_idx) {
                    ChunkResult {
                        chunk_idx: task.chunk_idx,
                        saved: false,
                        cache_file: None,
                        audit: ChunkAudit::empty(task.chunk_idx, "skip"),
                    }
                } else {
                    ChunkResult {
                        chunk_idx: task.chunk_idx,
                        saved: true,
                        cache_file: Some(std::path::PathBuf::from(format!(
                            "chunk_{}.parquet",
                            task.chunk_idx
                        ))),
                        audit: {
                            let mut a = ChunkAudit::empty(task.chunk_idx, "complete");
                            a.cache_file = Some(format!("chunk_{}.parquet", task.chunk_idx));
                            a.point_count = 100;
                            a.disturbance_count = 15;
                            a.background_count = 85;
                            a
                        },
                    }
                }
            })
            .collect();

        let skipped: Vec<&ChunkResult> = results.iter().filter(|r| !r.saved).collect();
        let processed: Vec<&ChunkResult> = results.iter().filter(|r| r.saved).collect();
        assert_eq!(skipped.len(), 5);
        assert_eq!(processed.len(), 5);
        assert!(processed.iter().all(|r| r.audit.disturbance_count == 15));
    }

    #[test]
    fn test_estimate_worker_memory_gb_1024() {
        let gb = estimate_worker_memory_gb(1024, 1024);
        assert!(gb > 0.0);
        assert!(gb < 1.0, "1024×1024 chunk should use < 1 GB, got {:.3}", gb);
    }

    #[test]
    fn test_estimate_worker_memory_gb_large() {
        let gb = estimate_worker_memory_gb(4096, 4096);
        assert!(gb > 0.5);
    }

    #[test]
    fn test_detect_system_ram() {
        let ram = detect_system_ram_gb();
        assert!(
            ram.is_some(),
            "should detect system RAM on all supported platforms"
        );
        assert!(ram.unwrap() > 0.0);
    }

    #[test]
    fn test_detect_available_ram() {
        let avail = detect_available_ram_gb();
        assert!(
            avail.is_some(),
            "should detect available RAM on all supported platforms"
        );
        assert!(avail.unwrap() > 0.0);
    }

    #[test]
    fn test_memory_guard_reduces_workers() {
        let est_gb = estimate_worker_memory_gb(256, 256);
        assert!(est_gb > 0.0, "256×256 should have non-zero memory estimate");

        let (count_limited, warning) =
            resolve_workers_with_memory_guard(8, 16, est_gb, Some(0.005), 0.7);
        assert_eq!(
            count_limited, 1,
            "0.005 GB limit with {:.4} GB/worker → 1",
            est_gb
        );
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("Reducing"));

        let (count_unlimited, _) = resolve_workers_with_memory_guard(8, 16, est_gb, None, 0.7);
        assert!(count_unlimited >= 1);
    }

    #[test]
    fn test_memory_guard_no_estimation_no_limit() {
        let (count, warning) = resolve_workers_with_memory_guard(4, 8, 0.0, None, 0.7);
        assert_eq!(count, 4);
        assert!(warning.is_none());
    }

    #[test]
    fn test_resolve_workers_defaults_to_at_least_one() {
        let (count, _) = resolve_workers_with_memory_guard(1, 1, 999.0, Some(0.001), 0.7);
        assert_eq!(count, 1);
    }
}
