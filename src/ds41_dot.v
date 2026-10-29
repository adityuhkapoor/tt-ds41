`timescale 1ns/1ps
// Block-scaled dot product for DeepSeek-V4.1 linears: FP4 x FP8 for routed
// experts (W_FP8=0) or FP8 x FP8 for attention and the shared expert
// (W_FP8=1). Consumes LANES elements per in_valid; every 32 elements close a
// block, which is scaled by 2^(w_scale-127) * 2^(a_scale-127) (both UE8M0)
// and added into acc. acc is fixed point with 40 fractional bits.
//
// w_scale / a_scale must be held stable on the beat that closes a block.
`default_nettype none

module ds41_dot #(
    parameter LANES = 1,  // must divide 32
    parameter W_FP8 = 0
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clear,
    input  wire                    in_valid,
    input  wire [(W_FP8 ? 8 : 4)*LANES-1:0] w,  // lane i in w[WB*i +: WB]
    input  wire [8*LANES-1:0]      a,        // lane i in a[8*i +: 8]
    input  wire [7:0]              w_scale,
    input  wire [7:0]              a_scale,
    output reg  signed [63:0]      acc
);
    `include "ds41_fp.vh"

    localparam WB = W_FP8 ? 8 : 4;

    localparam STEPS = 32 / LANES;
    // Block sums: FP4 x FP8 fits 28 bits, FP8 x FP8 fits 42. The shift
    // offset moves the product's fractional bits (10 or 18) to 40, and the
    // top clamp keeps the shifted sum inside 64 bits.
    localparam [9:0] OFFSET = W_FP8 ? 10'd232 : 10'd224;
    localparam [5:0] SMAX   = W_FP8 ? 6'd21 : 6'd30;

    reg        [5:0]  cnt;
    reg signed [47:0] bsum;
    reg signed [47:0] lsum;
    integer i;

    always @* begin
        lsum = 48'sd0;
        for (i = 0; i < LANES; i = i + 1)
            if (W_FP8) lsum = lsum + fp8_fix(w[WB*i +: 8]) * fp8_fix(a[8*i +: 8]);
            else       lsum = lsum + fp4_fix(w[WB*i +: 4]) * fp8_fix(a[8*i +: 8]);
    end

    wire signed [47:0] bnext = bsum + lsum;
    wire signed [9:0]  shamt = $signed({2'b0, w_scale}) + $signed({2'b0, a_scale}) - $signed(OFFSET);

    always @(posedge clk) begin
        if (!rst_n || clear) begin
            acc  <= 64'sd0;
            bsum <= 32'sd0;
            cnt  <= 6'd0;
        end else if (in_valid) begin
            if (cnt == STEPS - 1) begin
                acc  <= acc + block_shift(bnext, shamt, SMAX);
                bsum <= 48'sd0;
                cnt  <= 6'd0;
            end else begin
                bsum <= bnext;
                cnt  <= cnt + 6'd1;
            end
        end
    end
endmodule

`default_nettype wire
