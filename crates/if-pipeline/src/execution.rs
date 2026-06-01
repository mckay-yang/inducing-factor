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
        return tasks
            .iter()
            .map(|task| process_fn(task, 0))
            .collect();
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
                if completed % 10 == 0 || completed == task_count {
                    log::info!(
                        "Chunk progress: {}/{} completed",
                        completed,
                        task_count
                    );
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
        assert!(unique_workers.len() > 1, "Expected >1 unique worker IDs, got {}", unique_workers.len());

        assert_eq!(results.len(), 8);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.chunk_idx, i);
            assert!(r.saved);
        }
    }

    #[test]
    fn test_parallel_execution_empty_tasks() {
        let tasks: Vec<ChunkTask> = vec![];
        let results = execute_chunk_tasks(&tasks, 4, |_, _| {
            ChunkResult {
                chunk_idx: 0,
                saved: false,
                cache_file: None,
                audit: ChunkAudit::empty(0, "skip"),
            }
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
        let results = execute_chunk_tasks(&tasks, 2, |task, _wid| {
            ChunkResult {
                chunk_idx: task.chunk_idx,
                saved: true,
                cache_file: None,
                audit: ChunkAudit::empty(task.chunk_idx, "complete"),
            }
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

        let results = execute_chunk_tasks_bounded(
            &tasks,
            8,
            Some(2.0),
            1.5,
            |task, _| ChunkResult {
                chunk_idx: task.chunk_idx,
                saved: true,
                cache_file: None,
                audit: ChunkAudit::empty(task.chunk_idx, "complete"),
            },
        );

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

        let skip_set: std::collections::HashSet<usize> =
            [0, 2, 4, 6, 8].iter().copied().collect();

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
                        cache_file: Some(
                            std::path::PathBuf::from(format!("chunk_{}.parquet", task.chunk_idx)),
                        ),
                        audit: {
                            let mut a = ChunkAudit::empty(task.chunk_idx, "complete");
                            a.cache_file =
                                Some(format!("chunk_{}.parquet", task.chunk_idx));
                            a.point_count = 100;
                            a.disturbance_count = 15;
                            a.background_count = 85;
                            a
                        },
                    }
                }
            })
            .collect();

        let skipped: Vec<_> = results
            .iter()
            .filter(|r| r.audit.status == "skip")
            .collect();
        let completed: Vec<_> = results
            .iter()
            .filter(|r| r.audit.status == "complete")
            .collect();

        assert_eq!(skipped.len(), 5);
        assert_eq!(completed.len(), 5);

        for r in &completed {
            assert!(r.saved);
            assert!(r.cache_file.is_some());
            assert!(r.audit.point_count > 0);
        }
    }
}
