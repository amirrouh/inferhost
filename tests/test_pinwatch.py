"""Tests for the pinwatch warm-decision logic (pins_to_warm)."""
from inferhost.pinwatch import pins_to_warm

PIN = "qwen-pinned"
GUEST = "bonsai-guest"


def test_cold_pin_idle_gpu_is_warmed():
    assert pins_to_warm({}, [PIN]) == [PIN]


def test_resident_pin_needs_nothing():
    assert pins_to_warm({PIN: "ready"}, [PIN]) == []


def test_starting_pin_is_left_alone():
    assert pins_to_warm({PIN: "starting"}, [PIN]) == []


def test_resident_guest_is_never_preempted():
    assert pins_to_warm({GUEST: "ready"}, [PIN]) == []


def test_guest_mid_transition_blocks_warming():
    assert pins_to_warm({GUEST: "starting"}, [PIN]) == []
    assert pins_to_warm({GUEST: "stopping"}, [PIN]) == []


def test_stopped_guest_entry_does_not_block():
    # llama-swap keeps exited processes in /running with state=stopped; they
    # hold no VRAM, so the pin should come back.
    assert pins_to_warm({GUEST: "stopped"}, [PIN]) == [PIN]


def test_stopped_pin_entry_is_rewarmed():
    # A pinned llama-server that crashed shows up as state=stopped.
    assert pins_to_warm({PIN: "stopped"}, [PIN]) == [PIN]


def test_multiple_pins_only_cold_ones_warm():
    other = "second-pin"
    assert pins_to_warm({PIN: "ready"}, [PIN, other]) == [other]


def test_no_pins_no_work():
    assert pins_to_warm({GUEST: "ready"}, []) == []
    assert pins_to_warm({}, []) == []
