"""core/retry.with_retry の契約: 指数バックオフ・最終試行の例外・再試行の可否。"""

import pytest

from core.retry import with_retry


class Boom(Exception):
    pass


class Fatal(Exception):
    pass


def _flaky(fail_times, exc=Boom):
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise exc(f"fail {calls['n']}")
        return "ok"

    return fn, calls


def _retry_boom(e):
    return isinstance(e, Boom)


def test_success_first_try_no_sleep_no_callback(mocker):
    sleep = mocker.patch("time.sleep")
    on_retry = mocker.Mock()
    assert with_retry(lambda: 1, retries=3, is_retryable=_retry_boom, on_retry=on_retry) == 1
    sleep.assert_not_called()
    on_retry.assert_not_called()


def test_retries_then_succeeds_with_exponential_backoff(mocker):
    sleep = mocker.patch("time.sleep")
    fn, calls = _flaky(2)
    assert with_retry(fn, retries=3, is_retryable=_retry_boom) == "ok"
    assert calls["n"] == 3
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2]


def test_exhausted_reraises_last_exception_without_trailing_sleep(mocker):
    sleep = mocker.patch("time.sleep")
    seen = []
    fn, calls = _flaky(99)
    with pytest.raises(Boom, match="fail 3"):
        with_retry(fn, retries=3, is_retryable=_retry_boom, on_retry=lambda a, e: seen.append(a))
    assert calls["n"] == 3
    assert sleep.call_count == 2  # 最終試行の後は待たない
    assert seen == [0, 1]  # on_retry は最終試行では呼ばれない・0 始まり


def test_non_retryable_raises_immediately(mocker):
    sleep = mocker.patch("time.sleep")
    fn, calls = _flaky(99, exc=Fatal)
    with pytest.raises(Fatal):
        with_retry(fn, retries=3, is_retryable=_retry_boom)
    assert calls["n"] == 1
    sleep.assert_not_called()


def test_retries_one_means_single_attempt(mocker):
    sleep = mocker.patch("time.sleep")
    fn, calls = _flaky(99)
    with pytest.raises(Boom):
        with_retry(fn, retries=1, is_retryable=_retry_boom)
    assert calls["n"] == 1
    sleep.assert_not_called()


def test_zero_retries_is_rejected():
    with pytest.raises(ValueError):
        with_retry(lambda: 1, retries=0, is_retryable=_retry_boom)


def test_keyboard_interrupt_is_not_swallowed(mocker):
    mocker.patch("time.sleep")

    def fn():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        with_retry(fn, retries=3, is_retryable=lambda e: True)


def test_sleep_is_looked_up_at_call_time(mocker):
    """呼び出し側のテストが time.sleep を後から patch できる(デフォルト引数に束縛しない)。"""
    fn, _ = _flaky(1)
    sleep = mocker.patch("time.sleep")
    with_retry(fn, retries=2, is_retryable=_retry_boom)
    sleep.assert_called_once_with(1)
