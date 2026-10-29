# SPDX-FileCopyrightText: © 2026 Aditya K
# SPDX-License-Identifier: Apache-2.0
"""Drives tt_um_ds41_moe through its byte protocol and checks every accumulator byte against an
exact fixed-point model. The same test runs on the RTL and on the gate-level netlist."""

import random
from fractions import Fraction
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

FINITE = [b for b in range(256) if (b & 0x7F) != 0x7F]
OFFSET, SMAX = 224, 30


def fp4_fix(n):
    """FP4 e2m1 as a signed integer with 1 fractional bit."""
    e, m = (n >> 1) & 3, n & 1
    mag = m if e == 0 else (2 | m) << (e - 1)
    return -mag if n & 8 else mag


def fp8_fix(b):
    """FP8 e4m3 as a signed integer with 9 fractional bits."""
    e, m = (b >> 3) & 15, b & 7
    mag = m if e == 0 else (8 | m) << (e - 1)
    return -mag if b & 0x80 else mag


def acc_model(w, ws, a, as_):
    """The 64-bit accumulator (40 fractional bits) exactly as the RTL builds it."""
    acc = 0
    for k in range(len(as_)):
        bsum = sum(fp4_fix(w[i]) * fp8_fix(a[i]) for i in range(32 * k, 32 * k + 32))
        s = min(max(ws[k] + as_[k] - OFFSET, -63), SMAX)
        acc += bsum << s if s >= 0 else bsum >> -s
    return acc & (2**64 - 1)


def exact(w, ws, a, as_):
    """The true rational dot product, for checking the fixed point is lossless."""
    return sum(Fraction(fp4_fix(w[i]), 2) * Fraction(fp8_fix(a[i]), 512)
               * Fraction(2) ** (ws[i // 32] + as_[i // 32] - 254) for i in range(len(w)))


async def write(dut, sel, byte):
    dut.ui_in.value = byte
    dut.uio_in.value = sel << 1
    await ClockCycles(dut.clk, 2)
    dut.uio_in.value = sel << 1 | 1
    await ClockCycles(dut.clk, 4)
    dut.uio_in.value = sel << 1
    await ClockCycles(dut.clk, 4)


async def dot(dut, w, ws, a, as_):
    await write(dut, 0, 0x01)  # clear accumulator
    for k in range(len(as_)):
        await write(dut, 1, ws[k])
        await write(dut, 2, as_[k])
        for i in range(32 * k, 32 * k + 32, 2):
            await write(dut, 3, w[i] | w[i + 1] << 4)
            await write(dut, 4, a[i])
            await write(dut, 4, a[i + 1])
    await write(dut, 0, 0x02)  # rewind read pointer
    got = 0
    for k in range(8):
        got |= int(dut.uo_out.value) << (8 * k)
        await write(dut, 5, 0)
    return got


async def reset(dut):
    cocotb.start_soon(Clock(dut.clk, 40, unit="ns").start())  # 25 MHz, the hardened clock
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_random_dots(dut):
    await reset(dut)
    assert int(dut.uio_oe.value) == 0, "uio pins must all be inputs"
    rng = random.Random(41)
    for n in range(10):
        nblk = rng.randrange(1, 4)
        w = [rng.randrange(16) for _ in range(32 * nblk)]
        a = [rng.choice(FINITE) for _ in range(32 * nblk)]
        if n % 3 == 2:  # any scales, including the saturating and truncating shifts
            ws = [rng.randrange(256) for _ in range(nblk)]
            as_ = [rng.randrange(256) for _ in range(nblk)]
        else:  # the range real checkpoints use, where the result must be exact
            ws = [rng.randrange(112, 128) for _ in range(nblk)]
            as_ = [rng.randrange(112, 132) for _ in range(nblk)]
        want = acc_model(w, ws, a, as_)
        got = await dot(dut, w, ws, a, as_)
        dut._log.info(f"dot {n}: {nblk} blocks, acc {got:016x}")
        assert got == want, f"dot {n}: got {got:016x} want {want:016x}"
        if n % 3 != 2 and all(224 <= x + y <= 254 for x, y in zip(ws, as_)):
            signed = got - 2**64 if got >> 63 else got
            assert Fraction(signed, 2**40) == exact(w, ws, a, as_)


@cocotb.test()
async def test_real_expert_row(dut):
    """A full 5120-element row of a real DeepSeek-V4.1 routed expert against FP8 activations."""
    await reset(dut)
    lines = [l for l in (Path(__file__).parent / "expert_row.hex").read_text().split("\n") if l and l[0] != "#"]
    ws = list(bytes.fromhex(lines[0]))
    packed = bytes.fromhex(lines[1])
    w = [v for b in packed for v in (b & 15, b >> 4)]
    rng = random.Random(7)
    a = [rng.choice(FINITE) for _ in w]
    as_ = [rng.randrange(118, 126) for _ in ws]
    want = acc_model(w, ws, a, as_)
    got = await dot(dut, w, ws, a, as_)
    signed = got - 2**64 if got >> 63 else got
    dut._log.info(f"expert row: acc {got:016x} = {signed / 2**40:.6f}")
    assert got == want
    assert Fraction(signed, 2**40) == exact(w, ws, a, as_)
