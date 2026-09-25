# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""A software mock of the IRON ObjectFifo/Worker synchronization semantics.

This does not model timing, and it is not a substitute for hardware. It models
the one thing that makes AIE dataflow bugs expensive to find on device: the
*synchronization*. Bounded queues at the depth actually declared, acquire and
release paired the way the kernels pair them, broadcast objects freed only once
every consumer has released, and a timeout on every blocking operation so a
deadlock fails in seconds with a named FIFO instead of hanging a test run.

The kernels' math is plugged in as plain numpy, not stubs, so the same run that
catches a deadlock also catches an arithmetic bug.

On device, the corresponding failure modes are "it hangs" and "the output is
wrong", neither of which volunteers any further information.
"""

from __future__ import annotations

import threading

DEFAULT_TIMEOUT_S = 5.0


class DeadlockError(RuntimeError):
    """A blocking acquire/release did not make progress within its timeout.

    Carries the FIFO name and the counters at the moment it gave up, because
    "which FIFO starved, and how far had it got" is the whole diagnosis.
    """


class ObjectFifo:
    """A depth-bounded object FIFO with optional broadcast to several consumers.

    Mirrors the IRON semantics the kernels rely on:

      * a producer acquires a free slot, fills it, and releases it, which makes
        the object visible to every consumer;
      * each consumer acquires and releases independently, and a slot is only
        recycled once *all* consumers have released it (this is what makes a
        broadcast FIFO's depth behave differently from a single-consumer one);
      * a producer blocks while ``produced - min(consumed) == depth``.

    ``n_consumers`` > 1 is the broadcast case: the GEMM broadcasts each A tile
    across the columns of a row, and each B tile across the rows of a column.
    """

    def __init__(self, name: str, depth: int, n_consumers: int = 1) -> None:
        if depth < 1:
            msg = f"{name}: depth must be >= 1, got {depth}"
            raise ValueError(msg)
        if n_consumers < 1:
            msg = f"{name}: n_consumers must be >= 1, got {n_consumers}"
            raise ValueError(msg)
        self.name = name
        self.depth = depth
        self.n_consumers = n_consumers
        self._slots: list[object] = [None] * depth
        self._produced = 0
        self._consumed = [0] * n_consumers
        self._cv = threading.Condition()

    # -- introspection -----------------------------------------------------
    @property
    def produced(self) -> int:
        with self._cv:
            return self._produced

    def _state(self) -> str:
        return (
            f"{self.name}(depth={self.depth}, produced={self._produced}, "
            f"consumed={list(self._consumed)})"
        )

    def _wait(self, predicate, what: str, timeout: float) -> None:
        if not self._cv.wait_for(predicate, timeout=timeout):
            msg = f"{what} timed out after {timeout}s on {self._state()}"
            raise DeadlockError(msg)

    # -- producer side -----------------------------------------------------
    def acquire_produce(self, timeout: float = DEFAULT_TIMEOUT_S) -> int:
        """Block for a free slot; return its index. Does not publish it."""
        with self._cv:
            self._wait(
                lambda: self._produced - min(self._consumed) < self.depth,
                f"acquire_produce({self.name})",
                timeout,
            )
            return self._produced % self.depth

    def release_produce(self, obj: object) -> None:
        """Publish ``obj`` into the slot reserved by ``acquire_produce``."""
        with self._cv:
            self._slots[self._produced % self.depth] = obj
            self._produced += 1
            self._cv.notify_all()

    # -- consumer side -----------------------------------------------------
    def acquire_consume(
        self, consumer: int = 0, timeout: float = DEFAULT_TIMEOUT_S
    ) -> object:
        """Block until this consumer's next object exists; return it."""
        with self._cv:
            self._wait(
                lambda: self._consumed[consumer] < self._produced,
                f"acquire_consume({self.name}, consumer={consumer})",
                timeout,
            )
            return self._slots[self._consumed[consumer] % self.depth]

    def release_consume(self, consumer: int = 0) -> None:
        with self._cv:
            self._consumed[consumer] += 1
            self._cv.notify_all()


def run_workers(workers, timeout: float = 30.0) -> None:
    """Run worker callables as threads; re-raise the first failure.

    A DeadlockError raised inside any worker surfaces here, so a test that
    deadlocks fails with the starved FIFO named rather than hanging.
    """
    errors: list[BaseException] = []
    lock = threading.Lock()

    def wrap(fn):
        def run():
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                with lock:
                    errors.append(exc)

        return run

    threads = [threading.Thread(target=wrap(w), daemon=True) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)

    if errors:
        raise errors[0]

    stuck = [t for t in threads if t.is_alive()]
    if stuck:
        msg = f"{len(stuck)} worker(s) still running after {timeout}s (deadlock)"
        raise DeadlockError(msg)
