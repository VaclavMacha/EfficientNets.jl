struct EfficientNet{M,T}
    layers::T

    EfficientNet(M::Type{<:ModelName}, layers) = new{M,typeof(layers)}(layers)
end

@functor EfficientNet

function (m::EfficientNet)(input)
    return m.layers(input)
end

function Base.show(io::IO, m::EfficientNet{M}) where {M}
    print(io, "EfficientNet-", M, "(", m.layers, ")")
    return
end

function EfficientNet(
    T::Type{<:ModelName};
    n_classes::Integer=1000,
    input_channels::Integer=3,
    include_top::Bool=true,
    include_head::Bool=true,
    pretrained::Bool=false
)

    # model parameters
    bias = false
    pad = SamePad()
    width, depth, res, dropout = parameters(T)
    params = ModelParams((res, res), width, depth, n_classes, 8, 8, dropout, include_top)

    ch_in = input_channels
    ch_out = compute_channels(32, params)

    # MBConv blocks parameters
    blocks = (
        BlockParams{Float32,Int}(1, (3, 3), 32 => 16, 1, 1, 0.25, true, 0.99, 1.0e-3),
        BlockParams{Float32,Int}(2, (3, 3), 16 => 24, 2, 6, 0.25, true, 0.99, 1.0e-3),
        BlockParams{Float32,Int}(2, (5, 5), 24 => 40, 2, 6, 0.25, true, 0.99, 1.0e-3),
        BlockParams{Float32,Int}(3, (3, 3), 40 => 80, 2, 6, 0.25, true, 0.99, 1.0e-3),
        BlockParams{Float32,Int}(3, (5, 5), 80 => 112, 1, 6, 0.25, true, 0.99, 1.0e-3),
        BlockParams{Float32,Int}(4, (5, 5), 112 => 192, 2, 6, 0.25, true, 0.99, 1.0e-3),
        BlockParams{Float32,Int}(1, (3, 3), 192 => 320, 1, 6, 0.25, true, 0.99, 1.0e-3),
    )

    # head parameters
    ch_head_in = compute_channels(blocks[end].channels[2], params)
    ch_head_out = compute_channels(1280, params)

    # layers
    layers = Chain(
        # Stem
        Chain(
            Conv((3, 3), ch_in => ch_out; stride=2, bias, pad),
            BatchNorm(ch_out, swish),
        ),

        # MBConv blocks
        Chain(
            MBConvBlock.(blocks, Ref(params))...,
        ),

        # Head
        !include_head ? identity : Chain(
            Conv((1, 1), ch_head_in => ch_head_out; bias, pad),
            BatchNorm(ch_head_out, swish),
            AdaptiveMeanPool((1, 1)),
        ),

        # Top
        !include_top ? identity : Chain(
            Flux.flatten,
            Dense(ch_head_out, params.n_classes),
        ),
    )

    # create model and load pretrained weights
    model = EfficientNet(T, layers)
    if pretrained
        _load_weights!(model)
        @warn "pretrained models are not implemented yet"
    end
    return model
end
