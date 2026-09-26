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

    def __init__(
        self,
        name: str,
        depth: int,
        n_consumers: int = 1,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if depth < 1:
            msg = f"{name}: depth must be >= 1, got {depth}"
            raise ValueError(msg)
        if n_consumers < 1:
            msg = f"{name}: n_consumers must be >= 1, got {n_consumers}"
            raise ValueError(msg)
        self.name = name
        self.depth = depth
        # Default deadline for this FIFO's blocking operations. A test that
        # deliberately starves a FIFO sets it low so the expected failure is
        # fast rather than dominating the suite's wall time.
        self.timeout = timeout
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
    def acquire_produce(self, timeout: float | None = None) -> int:
        """Block for a free slot; return its index. Does not publish it."""
        timeout = self.timeout if timeout is None else timeout
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
    def acquire_consume(self, consumer: int = 0, timeout: float | None = None) -> object:
        """Block until this consumer's next object exists; return it."""
        timeout = self.timeout if timeout is None else timeout
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


def pad_to(value: int, multiple: int) -> int:
    """Round ``value`` up to a multiple of ``multiple`` (the host's GGML_PAD)."""
    return ((value + multiple - 1) // multiple) * multiple


def build_gemm_fifos(
    n_aie_rows: int, n_aie_cols: int, depth: int, timeout: float = DEFAULT_TIMEOUT_S
):
    """The FIFO grid the whole-array GEMM builds, shared by every model of it.

    A is broadcast across the columns of a row and B across the rows of a
    column, so those FIFOs have several consumers each; every core owns a
    private C FIFO. Returns ``(a_fifos, b_fifos, c_fifos)``.
    """
    a_fifos = [
        ObjectFifo(f"A_l2l1[{r}]", depth, n_consumers=n_aie_cols, timeout=timeout)
        for r in range(n_aie_rows)
    ]
    b_fifos = [
        ObjectFifo(f"B_l2l1[{c}]", depth, n_consumers=n_aie_rows, timeout=timeout)
        for c in range(n_aie_cols)
    ]
    c_fifos = [
        [ObjectFifo(f"C_l1l2[{r}][{c}]", depth, timeout=timeout) for c in range(n_aie_cols)]
        for r in range(n_aie_rows)
    ]
    return a_fifos, b_fifos, c_fifos


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
