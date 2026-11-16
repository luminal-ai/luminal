//! One caller at a time, whichever thread each caller is on.
//!
//! A compiled graph is an ordinary Python object: it is passed between
//! threads, called on any of them and freed on whichever drops it last.
//! What it holds is not safe to use from two calls at once, and nothing
//! here tries to make it so. [`Serialized`] lets one call in at a time and
//! makes every other caller wait.
//!
//! Two things keep the waiting safe. A caller waits detached from the
//! interpreter, so the call it waits for can go on using Python. And a call
//! that comes back in on its own thread, which could only wait for itself,
//! is refused instead.

use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;

/// State that one call uses at a time.
pub struct Serialized<T> {
    state: Mutex<T>,
    /// The thread that is inside a call, by its token; zero when none is.
    inside: AtomicUsize,
}

/// The state, for the length of one call.
pub struct Entered<'a, T> {
    // Declared first so that it is dropped first: the token is cleared
    // while the lock is still held. Cleared after the release, it could
    // erase the token of the caller that took the lock next.
    _inside: Inside<'a>,
    state: MutexGuard<'a, T>,
}

struct Inside<'a>(&'a AtomicUsize);

impl Drop for Inside<'_> {
    fn drop(&mut self) {
        self.0.store(0, Ordering::Release);
    }
}

/// A value no two live threads share: the address of a thread-local.
fn thread_token() -> usize {
    thread_local!(static TOKEN: u8 = const { 0 });
    TOKEN.with(|token| token as *const u8 as usize)
}

impl<T> Serialized<T> {
    pub fn new(state: T) -> Self {
        Self {
            state: Mutex::new(state),
            inside: AtomicUsize::new(0),
        }
    }

    /// Enter the state for one call, waiting for a call in progress on
    /// another thread. `what` names the object in the refusal of a call
    /// that re-enters it on its own thread.
    pub fn enter(&self, py: Python<'_>, what: &str) -> PyResult<Entered<'_, T>> {
        let me = thread_token();
        if self.inside.load(Ordering::Acquire) == me {
            return Err(PyRuntimeError::new_err(format!(
                "{what} was called again from inside its own call on the same thread; its \
                 calls run one at a time"
            )));
        }
        // A call that panicked left the state as it was at the panic. Every
        // call restates what it reads, so the next call goes ahead.
        let state = self
            .state
            .lock_py_attached(py)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.inside.store(me, Ordering::Release);
        Ok(Entered {
            _inside: Inside(&self.inside),
            state,
        })
    }
}

impl<T> Deref for Entered<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.state
    }
}

impl<T> DerefMut for Entered<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.state
    }
}
