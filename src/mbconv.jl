struct MBConv{T}
    layers::T
end

@functor MBConv

function (m::MBConv)(input)
    return m.layers(input)
end

function Base.show(io::IO, m::MBConv)
    print(io, "MBConv(", m.layers, ")")
    return
end

"""
    MBConv(kernel, in => out; stride, pad, bias, skip_connection, expansion_ratio, se_ratio)

Mobile Inverted Residual Bottleneck Block.

# Arguments:
- `filter`: Size of the kernel for the depthwise conv phase.
- `in`: Number of input channels.
- `out`: Number of output channels.

# Keyword arguments
- `stride = 1`: Size of the stride for the depthwise conv phase.
- `expand_ratio = 1`: Expansion ratio defines the number of output channels `out_channels = input_channels * expansion_ratio`.
- `squeeze_ratio = 0`: Squeeze-Excitation ratio. Should be in `(0, 1]` range.
- `skip_connection = false`: Whether to use skip connection and drop connect.
- `dropout = 0`: Dropout rate. Should be in `(0, 1)` range.

# References:
1. https://arxiv.org/abs/1704.04861 (MobileNet v1)
2. https://arxiv.org/abs/1801.04381 (MobileNet v2)
3. https://arxiv.org/abs/1905.02244 (MobileNet v3)
"""
function MBConv(
    filter::NTuple{N,Integer},
    channels::Pair{<:Integer,<:Integer};
    stride=1,
    expand_ratio=1,
    squeeze_ratio=0,
    skip_connection=false,
    dropout=0,
    momentum=0.1f0,
    ϵ=1.0f-5
) where {N}

    ch_in, ch_out = channels
    ch_mid = ceil(Int, ch_in * expand_ratio)
    ch_sqz = max(1, ceil(Int, ch_in * squeeze_ratio))

    bias = false
    pad = SamePad()

    # layers
    layers = Chain(
        # Expansion phase (Inverted Bottleneck)
        expand_ratio == 1 ? identity : Chain(
            Conv((1, 1), ch_in => ch_mid; bias, pad),
            BatchNorm(ch_mid, swish; ϵ, momentum)
        ),

        # Depthwise convolution phase
        Chain(
            Conv(filter, ch_mid => ch_mid; groups=ch_mid, stride, bias, pad),
            BatchNorm(ch_mid, swish; ϵ, momentum),
        ),

        # Squeeze and Excitation layer
        !(0 < squeeze_ratio <= 1) ? identity : SkipConnection(
            Chain(
                AdaptiveMeanPool((1, 1)),
                Conv((1, 1), ch_mid => ch_sqz, swish; pad),
                Conv((1, 1), ch_sqz => ch_mid; pad),
            ),
            (mx, x) -> sigmoid.(mx) .* x
        ),

        # Pointwise convolution phase
        Chain(
            Conv((1, 1), ch_mid => ch_out; bias, pad),
            BatchNorm(ch_out; ϵ, momentum),
        ),
    )

    # Skip connection and drop connect
    if skip_connection && stride == 1 && ch_in == ch_out
        if 0 < dropout < 1
            layers = Chain(
                layers.layers...,
                Dropout(dropout),
            )
        end
        layers = SkipConnection(layers, +)
    end
    return MBConv(layers)
end

function compute_channels(channels_orig::Integer, params::ModelParams)
    multiplier = params.width_coef
    depth_divisor = params.depth_divisor

    if multiplier ≈ 1
        return channels_orig
    end

    channels = multiplier * channels_orig
    channels_new = max(
        params.depth_min,
        (floor(Int, channels + depth_divisor / 2) ÷ depth_divisor) * depth_divisor
    )
    if channels_new < 0.9 * channels
        return channels_new + depth_divisor
    else
        return channels_new
    end
end

function MBConvBlock(bparams::BlockParams, params::ModelParams)
    channels_in = compute_channels(bparams.channels[1], params)
    channels_out = compute_channels(bparams.channels[2], params)
    repeats = ceil(Int, bparams.repeats * params.depth_coef)

    return map(1:repeats) do i
        ch_in = i == 1 ? channels_in : channels_out
        ch_out = channels_out
        stride = i == 1 ? bparams.stride : 1

        return MBConv(
            bparams.filters,
            ch_in => ch_out;
            stride=stride,
            expand_ratio=bparams.expand_ratio,
            squeeze_ratio=bparams.squeeze_ratio,
            skip_connection=bparams.skip_connection,
            dropout=params.dropout,
            momentum=bparams.momentum,
            ϵ=bparams.ϵ
        )
    end
end
