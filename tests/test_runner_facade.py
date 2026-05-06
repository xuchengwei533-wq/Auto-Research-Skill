from __future__ import annotations

from nightrunner.runner import (
    apply_experiment,
    check_auth,
    clean,
    init_project,
    run,
    run_baseline,
    run_night,
    setup,
    status,
    tail,
)


def test_runner_public_functions_remain_importable() -> None:
    for fn in [
        init_project,
        setup,
        run_baseline,
        run_night,
        run,
        apply_experiment,
        clean,
        status,
        tail,
        check_auth,
    ]:
        assert callable(fn)
