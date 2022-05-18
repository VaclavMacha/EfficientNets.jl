struct MBConv
    layers
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
- `drop_rate = 0`: Dropout rate. Should be in `(0, 1)` range.

# References:
1. https://arxiv.org/abs/1704.04861 (MobileNet v1)
2. https://arxiv.org/abs/1801.04381 (MobileNet v2)
3. https://arxiv.org/abs/1905.02244 (MobileNet v3)
"""
function MBConv(
    filter::NTuple{N,Integer},
    channels::Pair{<:Integer,<:Integer};
    stride=1,
    pad=SamePad(),
    bias=false,
    ϵ=1.0f-5,
    momentum=0.1f0,
    expand_ratio=1,
    squeeze_ratio=0,
    skip_connection=false,
    drop_rate=0
) where {N}

    channels_in, channels_out = channels
    channels_mid = ceil(Int, channels_in * expand_ratio)
    channels_squeezed = max(1, ceil(Int, channels_in * squeeze_ratio))

    # layers
    layers = Chain(
        # Expansion phase (Inverted Bottleneck)
        expand_ratio == 1 ? identity : Chain(
            Conv((1, 1), channels_in => channels_mid; bias, pad),
            BatchNorm(channels_mid, swish; ϵ, momentum)
        ),

        # Depthwise convolution phase
        Chain(
            Conv(filter, channels_mid => channels_mid; groups=channels_mid, bias, stride, pad),
            BatchNorm(channels_mid, swish; ϵ, momentum),
        ),

        # Squeeze and Excitation layer
        !(0 < squeeze_ratio <= 1) ? identity : SkipConnection(
            Chain(
                AdaptiveMeanPool((1, 1)),
                Conv((1, 1), channels_mid => channels_squeezed, swish; pad),
                Conv((1, 1), channels_squeezed => channels_mid; pad),
            ),
            (mx, x) -> sigmoid.(mx) .* x
        ),

        # Pointwise convolution phase
        Chain(
            Conv((1, 1), channels_mid => channels_out; pad, bias),
            BatchNorm(channels_out; ϵ, momentum),
        ),
    )

    # Skip connection and drop connect
    if skip_connection && stride == 1 && channels_in == channels_out
        if 0 < drop_rate < 1
            layers = Chain(
                layers.layers...,
                Dropout(drop_rate),
            )
        end
        layers = SkipConnection(layers, +)
    end
    return MBConv(layers)
end
