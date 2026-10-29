<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This tile is the arithmetic core of a DeepSeek-V4.1-Flash routed expert. It computes a dot product between FP4 e2m1
weights and FP8 e4m3 activations, both in the model's native block-scaled format. Each block of 32 elements shares one
UE8M0 (power-of-two) weight scale and one activation scale.

Nothing is rounded. Each FP4 weight becomes a small signed integer (x2) and each FP8 activation becomes a signed integer
(x512, subnormals included). Their 32 products sum exactly into a block total. The total is shifted by
`w_scale + a_scale - 224` into a 64-bit accumulator with 40 fractional bits. For the scale range real checkpoints use
(combined scale 224 to 254), the accumulator equals the exact rational dot product. Above that range the shift
saturates at 30, and below it the low bits are truncated.

The host writes one byte at a time. It puts the byte on `ui_in` and a register select on `uio[3:1]`, then pulses
`uio[0]`. The strobe is synchronized and edge-detected, so the RP2040 on the demo board can bit-bang it.

| sel | Register | Byte |
|-----|----------|------|
| 0 | CMD | bit 0 clears the accumulator, bit 1 rewinds the read pointer |
| 1 | WSCALE | UE8M0 weight block scale |
| 2 | ASCALE | UE8M0 activation block scale |
| 3 | WEIGHT | two FP4 weights, low nibble first |
| 4 | ACT | one FP8 activation, multiplied with the next weight nibble |
| 5 | NEXT | advances the read pointer |

`uo_out` always shows byte `[ptr]` of the accumulator, least significant byte first.

## How to test

For each 32-element block:

1. Write WSCALE and ASCALE.
2. Repeat 16 times: write one WEIGHT byte, then two ACT bytes.

Then write CMD `0x02` and read the 8 accumulator bytes from `uo_out`, writing NEXT between reads. The signed 64-bit
result divided by 2^40 is the dot product. Write CMD `0x01` to clear before the next dot product.

Hold the strobe high for at least 2 clocks and low for at least 2 clocks between writes. The cocotb test in `test/`
runs a full 5120-element row of a real DeepSeek-V4.1-Flash-1017 expert (layer 10, expert 7, `w1`) through the tile
and checks every bit against the exact result.

## External hardware

None. The RP2040 on the Tiny Tapeout demo board drives the pins.
