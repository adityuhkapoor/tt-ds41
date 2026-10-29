`timescale 1ns/1ps
// Tiny Tapeout tile: one-lane DeepSeek-V4.1 expert dot-product engine.
//
// The host (the TT demo board's RP2040, or a PC through it) streams weights
// and activations one byte at a time. Each write is latched on a rising edge
// of uio_in[0]; uio_in[3:1] selects the register, ui_in carries the byte.
//
//   sel 0  CMD      bit0 = clear accumulator, bit1 = rewind read pointer
//   sel 1  WSCALE   UE8M0 weight block scale
//   sel 2  ASCALE   UE8M0 activation block scale
//   sel 3  WEIGHT   two FP4 weights, low nibble first
//   sel 4  ACT      one FP8 e4m3 activation, MACed with the next weight nibble
//   sel 5  NEXT     advance read pointer
//
// uo_out always shows byte [ptr] of the 64-bit accumulator (LSB first).
`default_nettype none

module tt_um_ds41_moe (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);
    assign uio_out = 8'd0;
    assign uio_oe  = 8'd0;

    reg [2:0] strobe_sync;
    always @(posedge clk)
        if (!rst_n) strobe_sync <= 3'd0;
        else        strobe_sync <= {strobe_sync[1:0], uio_in[0]};
    wire wr = strobe_sync[1] & ~strobe_sync[2];

    wire [2:0] sel = uio_in[3:1];

    reg [7:0] w_scale, a_scale, wbyte;
    reg       nib_hi;
    reg [2:0] ptr;

    // An ACT write latches its operands here and the MAC runs on the next
    // clock, so no pin-to-register path goes through the multiplier.
    reg [3:0] mac_w;
    reg [7:0] mac_a;
    reg       mac;

    always @(posedge clk) begin
        if (!rst_n) begin
            w_scale <= 8'd0;
            a_scale <= 8'd0;
            wbyte   <= 8'd0;
            nib_hi  <= 1'b0;
            ptr     <= 3'd0;
            mac     <= 1'b0;
        end else begin
            mac <= wr && sel == 3'd4;
            if (wr) begin
                case (sel)
                    3'd0: if (ui_in[1]) ptr <= 3'd0;
                    3'd1: w_scale <= ui_in;
                    3'd2: a_scale <= ui_in;
                    3'd3: begin wbyte <= ui_in; nib_hi <= 1'b0; end
                    3'd4: begin
                        mac_w  <= nib_hi ? wbyte[7:4] : wbyte[3:0];
                        mac_a  <= ui_in;
                        nib_hi <= ~nib_hi;
                    end
                    3'd5: ptr <= ptr + 3'd1;
                    default: ;
                endcase
            end
        end
    end

    wire signed [63:0] acc;

    ds41_dot #(.LANES(1), .PIPE(1)) core (
        .clk      (clk),
        .rst_n    (rst_n),
        .clear    (wr && sel == 3'd0 && ui_in[0]),
        .in_valid (mac),
        .w        (mac_w),
        .a        (mac_a),
        .w_scale  (w_scale),
        .a_scale  (a_scale),
        .acc      (acc)
    );

    assign uo_out = acc[8*ptr +: 8];

    wire _unused = &{ena, uio_in[7:4], 1'b0};
endmodule

`default_nettype wire
