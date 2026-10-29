// Number formats used by DeepSeek-V4.1-Flash routed experts.
// Included inside a module body so the functions are local to it.
//
// Fixed-point conventions (all exact, no rounding):
//   fp4 e2m1 weight     -> signed int, 1 fractional bit  (x2),   |v| <= 12
//   fp8 e4m3 value      -> signed int, 9 fractional bits (x512), |v| <= 245760
//   fp4 x fp8 product   -> 10 fractional bits
//   fp8 x fp8 product   -> 18 fractional bits

function signed [7:0] fp4_fix;
    input [3:0] n;
    reg [4:0] mag;
    begin
        if (n[2:1] == 2'd0) mag = {4'd0, n[0]};
        else                mag = {3'd0, 1'b1, n[0]} << (n[2:1] - 2'd1);
        fp4_fix = n[3] ? -$signed({3'd0, mag}) : $signed({3'd0, mag});
    end
endfunction

function signed [19:0] fp8_fix;
    input [7:0] b;
    reg [18:0] mag;
    begin
        if (b[6:3] == 4'd0) mag = {16'd0, b[2:0]};
        else                mag = {15'd0, 1'b1, b[2:0]} << (b[6:3] - 4'd1);
        fp8_fix = b[7] ? -$signed({1'b0, mag}) : $signed({1'b0, mag});
    end
endfunction

// Scale a 32-element block sum into the accumulator's fixed point
// (40 fractional bits). The caller computes s from both UE8M0 scales and
// clamps the top end to what its block sum can take without overflow.
function signed [63:0] block_shift;
    input signed [47:0] v;
    input signed [9:0]  s;
    input        [5:0]  smax;
    reg signed [63:0] x;
    begin
        x = v;
        if (s > $signed({4'd0, smax})) block_shift = x <<< smax;
        else if (s >= 10'sd0)          block_shift = x <<< s;
        else if (s < -10'sd63)         block_shift = x >>> 63;
        else                           block_shift = x >>> (-s);
    end
endfunction
