use super::AbortOnDropHandle;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio::task::{spawn_local, JoinHandle};

/// A cloneable handle to a local pool, used for spawning `!Send` tasks.
///
/// Internally the local pool uses a [`tokio::runtime::LocalRuntime`] for each worker thread
/// in the pool. Consequently you can also use [`tokio::task::spawn_local`] (which will
/// execute on the same thread) inside the Future you supply to the various spawn methods
/// of `LocalPoolHandle`.
///
/// A worker remains alive while tasks spawned through the pool are in flight,
/// even after the pool handles have been dropped.
///
/// [`tokio::runtime::LocalRuntime`]: tokio::runtime::LocalRuntime
/// [`tokio::task::spawn_local`]: tokio::task::spawn_local
///
/// # Examples
///
/// ```
/// # #[cfg(not(target_family = "wasm"))]
/// # {
/// use std::rc::Rc;
/// use tokio::task;
/// use tokio_util::task::LocalPoolHandle;
///
/// #[tokio::main(flavor = "current_thread")]
/// async fn main() {
///     let pool = LocalPoolHandle::new(5);
///
///     let output = pool.spawn_pinned(|| {
///         // `data` is !Send + !Sync
///         let data = Rc::new("local data");
///         let data_clone = data.clone();
///
///         async move {
///             task::spawn_local(async move {
///                 println!("{}", data_clone);
///             });
///
///             data.to_string()
///         }
///     }).await.unwrap();
///     println!("output: {}", output);
/// }
/// # }
/// ```
///
#[derive(Clone)]
pub struct LocalPoolHandle {
    pool: Arc<LocalPool>,
}

impl LocalPoolHandle {
    /// Create a new pool of threads to handle `!Send` tasks. Spawn tasks onto this
    /// pool via [`LocalPoolHandle::spawn_pinned`].
    ///
    /// # Panics
    ///
    /// Panics if the pool size is less than one.
    #[track_caller]
    pub fn new(pool_size: usize) -> LocalPoolHandle {
        assert!(pool_size > 0);

        let workers = (0..pool_size)
            .map(|_| LocalWorkerHandle::new_worker())
            .collect();

        let pool = Arc::new(LocalPool { workers });

        LocalPoolHandle { pool }
    }

    /// Returns the number of threads of the Pool.
    #[inline]
    pub fn num_threads(&self) -> usize {
        self.pool.workers.len()
    }

    /// Returns the number of tasks scheduled on each worker. The indices of the
    /// worker threads correspond to the indices of the returned `Vec`.
    pub fn get_task_loads_for_each_worker(&self) -> Vec<usize> {
        self.pool
            .workers
            .iter()
            .map(|worker| worker.task_count.load(Ordering::SeqCst))
            .collect::<Vec<_>>()
    }

    /// Spawn a task onto a worker thread and pin it there so it can't be moved
    /// off of the thread. Note that the future is not [`Send`], but the
    /// [`FnOnce`] which creates it is.
    ///
    /// # Examples
    /// ```
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use std::rc::Rc;
    /// use tokio_util::task::LocalPoolHandle;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     // Create the local pool
    ///     let pool = LocalPoolHandle::new(1);
    ///
    ///     // Spawn a !Send future onto the pool and await it
    ///     let output = pool
    ///         .spawn_pinned(|| {
    ///             // Rc is !Send + !Sync
    ///             let local_data = Rc::new("test");
    ///
    ///             // This future holds an Rc, so it is !Send
    ///             async move { local_data.to_string() }
    ///         })
    ///         .await
    ///         .unwrap();
    ///
    ///     assert_eq!(output, "test");
    /// }
    /// # }
    /// ```
    pub fn spawn_pinned<F, Fut>(&self, create_task: F) -> JoinHandle<Fut::Output>
    where
        F: FnOnce() -> Fut,
        F: Send + 'static,
        Fut: Future + 'static,
        Fut::Output: Send + 'static,
    {
        self.pool
            .spawn_pinned(create_task, WorkerChoice::LeastBurdened)
    }

    /// Differs from `spawn_pinned` only in that you can choose a specific worker thread
    /// of the pool, whereas `spawn_pinned` chooses the worker with the smallest
    /// number of tasks scheduled.
    ///
    /// A worker thread is chosen by index. Indices are 0 based and the largest index
    /// is given by `num_threads() - 1`
    ///
    /// # Panics
    ///
    /// This method panics if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// This method can be used to spawn a task on all worker threads of the pool:
    ///
    /// ```
    /// # #[cfg(not(target_family = "wasm"))]
    /// # {
    /// use tokio_util::task::LocalPoolHandle;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     const NUM_WORKERS: usize = 3;
    ///     let pool = LocalPoolHandle::new(NUM_WORKERS);
    ///     let handles = (0..pool.num_threads())
    ///         .map(|worker_idx| {
    ///             pool.spawn_pinned_by_idx(
    ///                 || {
    ///                     async {
    ///                         "test"
    ///                     }
    ///                 },
    ///                 worker_idx,
    ///             )
    ///         })
    ///         .collect::<Vec<_>>();
    ///
    ///     for handle in handles {
    ///         handle.await.unwrap();
    ///     }
    /// }
    /// # }
    /// ```
    ///
    #[track_caller]
    pub fn spawn_pinned_by_idx<F, Fut>(&self, create_task: F, idx: usize) -> JoinHandle<Fut::Output>
    where
        F: FnOnce() -> Fut,
        F: Send + 'static,
        Fut: Future + 'static,
        Fut::Output: Send + 'static,
    {
        self.pool
            .spawn_pinned(create_task, WorkerChoice::ByIdx(idx))
    }
}

impl Debug for LocalPoolHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("LocalPoolHandle")
    }
}

enum WorkerChoice {
    LeastBurdened,
    ByIdx(usize),
}

struct LocalPool {
    workers: Box<[LocalWorkerHandle]>,
}

impl LocalPool {
    /// Spawn a `?Send` future onto a worker
    #[track_caller]
    fn spawn_pinned<F, Fut>(
        &self,
        create_task: F,
        worker_choice: WorkerChoice,
    ) -> JoinHandle<Fut::Output>
    where
        F: FnOnce() -> Fut,
        F: Send + 'static,
        Fut: Future + 'static,
        Fut::Output: Send + 'static,
    {
        let (worker, job_guard) = match worker_choice {
            WorkerChoice::LeastBurdened => self.find_and_incr_least_burdened_worker(),
            WorkerChoice::ByIdx(idx) => self.find_worker_by_idx(idx),
        };
        let worker_guard = worker.worker_guard.clone();

        worker.runtime_handle.spawn(async move {
            let _job_guard = job_guard;
            let _worker_guard = worker_guard;

            // A task running on the worker's LocalRuntime can spawn locally
            // without dispatching a separate creation callback.
            let mut join_handle =
                AbortOnDropHandle::new(spawn_local(async move { create_task().await }));
            let join_result = (&mut join_handle).await;
            drop(join_handle.detach());

            match join_result {
                Ok(output) => output,
                Err(e) => {
                    if e.is_panic() {
                        std::panic::resume_unwind(e.into_panic());
                    } else if e.is_cancelled() {
                        // No one else should have the join handle, so this is
                        // unexpected. Forward this error as a panic in the join
                        // handle.
                        panic!("spawn_pinned task was canceled: {e}");
                    } else {
                        // Something unknown happened (not a panic or
                        // cancellation). Forward this error as a panic in the
                        // join handle.
                        panic!("spawn_pinned task failed: {e}");
                    }
                }
            }
        })
    }

    /// Find the worker with the least number of tasks, increment its task
    /// count, and return its handle. Make sure to actually spawn a task on
    /// the worker so the task count is kept consistent with load.
    ///
    /// A job count guard is also returned to ensure the task count gets
    /// decremented when the job is done.
    fn find_and_incr_least_burdened_worker(&self) -> (&LocalWorkerHandle, JobCountGuard) {
        loop {
            let (worker, task_count) = self
                .workers
                .iter()
                .map(|worker| (worker, worker.task_count.load(Ordering::SeqCst)))
                .min_by_key(|&(_, count)| count)
                .expect("There must be more than one worker");

            // Make sure the task count hasn't changed since when we choose this
            // worker. Otherwise, restart the search.
            if worker
                .task_count
                .compare_exchange(
                    task_count,
                    task_count + 1,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return (worker, JobCountGuard(Arc::clone(&worker.task_count)));
            }
        }
    }

    #[track_caller]
    fn find_worker_by_idx(&self, idx: usize) -> (&LocalWorkerHandle, JobCountGuard) {
        let worker = &self.workers[idx];
        worker.task_count.fetch_add(1, Ordering::SeqCst);

        (worker, JobCountGuard(Arc::clone(&worker.task_count)))
    }
}

/// Automatically decrements a worker's job count when a job finishes (when
/// this gets dropped).
struct JobCountGuard(Arc<AtomicUsize>);

impl Drop for JobCountGuard {
    fn drop(&mut self) {
        // Decrement the job count
        let previous_value = self.0.fetch_sub(1, Ordering::SeqCst);
        debug_assert!(previous_value >= 1);
    }
}

struct LocalWorkerHandle {
    runtime_handle: tokio::runtime::Handle,
    worker_guard: Arc<oneshot::Sender<()>>,
    task_count: Arc<AtomicUsize>,
}

impl LocalWorkerHandle {
    /// Create a new worker for executing pinned tasks
    fn new_worker() -> LocalWorkerHandle {
        let (worker_guard, shutdown_receiver) = oneshot::channel();
        let (handle_sender, handle_receiver) = std::sync::mpsc::channel();

        let task_count = Arc::new(AtomicUsize::new(0));

        std::thread::spawn(|| Self::run(handle_sender, shutdown_receiver));

        let runtime_handle = handle_receiver
            .recv()
            .expect("Failed to recv local runtime init result")
            .expect("Failed to start local runtime");

        LocalWorkerHandle {
            runtime_handle,
            worker_guard: Arc::new(worker_guard),
            task_count,
        }
    }

    fn run(
        handle_sender: std::sync::mpsc::Sender<std::io::Result<tokio::runtime::Handle>>,
        shutdown_receiver: oneshot::Receiver<()>,
    ) {
        let runtime = match tokio::runtime::LocalRuntime::new() {
            Ok(runtime) => runtime,
            Err(err) => {
                let _ = handle_sender.send(Err(err));
                return;
            }
        };

        let runtime_handle = runtime.handle().clone();

        handle_sender
            .send(Ok(runtime_handle))
            .expect("Failed to send local runtime handle");
        drop(handle_sender);

        // The sender is shared by the pool and its in-flight proxy tasks.
        // Dropping the last owner signals that the worker can shut down.
        let _ = runtime.block_on(shutdown_receiver);

        // Give ready tasks spawned directly on the worker one last turn before
        // cancelling any that are still pending.
        runtime.block_on(tokio::task::yield_now());

        // Cancel any remaining tasks spawned directly on the worker.
        drop(runtime);
    }
}
