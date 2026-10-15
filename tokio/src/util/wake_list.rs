use core::mem::MaybeUninit;
use core::ptr;
use std::any::Any;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::task::Waker;

const NUM_WAKERS: usize = 32;

/// A list of wakers to be woken.
///
/// # Invariants
///
/// The first `curr` elements of `inner` are initialized.
pub(crate) struct WakeList {
    inner: [MaybeUninit<Waker>; NUM_WAKERS],
    curr: usize,
}

/// Hold wake panics until the caller has restored its bookkeeping.
#[derive(Default)]
pub(crate) struct WakeListPanic {
    first: Option<Box<dyn Any + Send>>,
    // Retain later payloads too: dropping arbitrary user data may itself panic.
    others: Vec<Box<dyn Any + Send>>,
}

impl WakeListPanic {
    fn capture(&mut self, panic: Box<dyn Any + Send>) {
        if self.first.is_none() {
            self.first = Some(panic);
        } else {
            self.others.push(panic);
        }
    }

    pub(crate) fn resume(self) {
        drop(self.others);
        if let Some(panic) = self.first {
            resume_unwind(panic);
        }
    }
}

impl WakeList {
    pub(crate) fn new() -> Self {
        const UNINIT_WAKER: MaybeUninit<Waker> = MaybeUninit::uninit();

        Self {
            inner: [UNINIT_WAKER; NUM_WAKERS],
            curr: 0,
        }
    }

    #[inline]
    pub(crate) fn can_push(&self) -> bool {
        self.curr < NUM_WAKERS
    }

    pub(crate) fn push(&mut self, val: Waker) {
        debug_assert!(self.can_push());

        self.inner[self.curr] = MaybeUninit::new(val);
        self.curr += 1;
    }

    #[cfg(any(
        feature = "net",
        feature = "process",
        feature = "sync",
        feature = "rt",
        feature = "signal",
        feature = "time",
    ))]
    pub(crate) fn wake_all(&mut self) {
        let mut panics = WakeListPanic::default();
        self.wake_all_defer_panic(&mut panics);
        panics.resume();
    }

    pub(crate) fn wake_all_defer_panic(&mut self, panics: &mut WakeListPanic) {
        struct DropGuard {
            start: *mut Waker,
            end: *mut Waker,
        }

        impl Drop for DropGuard {
            fn drop(&mut self) {
                // SAFETY: Both pointers are part of the same object, with `start <= end`.
                let len = unsafe { self.end.offset_from(self.start) } as usize;
                let slice = ptr::slice_from_raw_parts_mut(self.start, len);
                // SAFETY: All elements in `start..len` are initialized, so we can drop them.
                unsafe { ptr::drop_in_place(slice) };
            }
        }

        debug_assert!(self.curr <= NUM_WAKERS);

        let mut guard = {
            let start = self.inner.as_mut_ptr().cast::<Waker>();
            // SAFETY: The resulting pointer is in bounds or one after the length of the same object.
            let end = unsafe { start.add(self.curr) };
            // Transfer ownership of the wakers in `inner` to `DropGuard`.
            self.curr = 0;
            DropGuard { start, end }
        };
        while !ptr::eq(guard.start, guard.end) {
            // SAFETY: `start` is always initialized if `start != end`.
            let waker = unsafe { ptr::read(guard.start) };
            // SAFETY: The resulting pointer is in bounds or one after the length of the same object.
            guard.start = unsafe { guard.start.add(1) };
            if let Err(panic) = catch_unwind(AssertUnwindSafe(|| waker.wake())) {
                panics.capture(panic);
            }
        }
    }
}

impl Drop for WakeList {
    fn drop(&mut self) {
        let slice =
            ptr::slice_from_raw_parts_mut(self.inner.as_mut_ptr().cast::<Waker>(), self.curr);
        // SAFETY: The first `curr` elements are initialized, so we can drop them.
        unsafe { ptr::drop_in_place(slice) };
    }
}
