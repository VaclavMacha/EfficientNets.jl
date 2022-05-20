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

stages(::EfficientNet{M}) where {M} = stages(M)
stages_channels(::EfficientNet{M}) where {M} = stages_channels(M)

function EfficientNet(
    T::Type{<:ModelName};
    classes::Integer=1000,
    channels::Integer=3,
    top::Bool=true,
    head::Bool=true,
    pretrained::Bool=false,
    adversarial::Bool=false,
    exact::Bool = true,
)

    # check if pretrained model exists
    if pretrained && channels != 3
        msg = "Pre-trained models are available only for `channels = 3`."
        throw(ArgumentError(msg))
    end
    if pretrained && classes != 1000 && top
        if exact
            msg = "Pre-trained models are available only for `classes = 1000`. If you want to use randomly generated last layer with different number of output classes, use `exact = false`."
            throw(ArgumentError(msg))
        else
            @warn "Pre-trained models are available only for `classes = 1000`. Last layer of returned model is randomly generated."
        end
    end
    # check
    if pretrained
        path = load_path(T; adversarial)
    end

    # build random model
    model = EfficientNet(T, efficientnet(T, classes, channels, top, head))

    # load pre-trained weights
    pretrained && _load!(model, path; load_top = classes == 1000)
    return model
end

function efficientnet(
    T::Type{<:ModelName},
    classes::Integer,
    channels::Integer,
    top::Bool,
    head::Bool,
)

    # model parameters
    bias = false
    pad = SamePad()
    width, depth, res, dropout = parameters(T)
    params = ModelParams((res, res), width, depth, 8, 8, dropout, top)
    ch_out = compute_channels(32, params)

    # MBConv blocks parameters
    blocks = (
        BlockParams{Float32,Int}(1, (3, 3), 32 => 16, 1, 1, 0.25, true, 0.01f0, 1.0f-3),
        BlockParams{Float32,Int}(2, (3, 3), 16 => 24, 2, 6, 0.25, true, 0.01f0, 1.0f-3),
        BlockParams{Float32,Int}(2, (5, 5), 24 => 40, 2, 6, 0.25, true, 0.01f0, 1.0f-3),
        BlockParams{Float32,Int}(3, (3, 3), 40 => 80, 2, 6, 0.25, true, 0.01f0, 1.0f-3),
        BlockParams{Float32,Int}(3, (5, 5), 80 => 112, 1, 6, 0.25, true, 0.01f0, 1.0f-3),
        BlockParams{Float32,Int}(4, (5, 5), 112 => 192, 2, 6, 0.25, true, 0.01f0, 1.0f-3),
        BlockParams{Float32,Int}(1, (3, 3), 192 => 320, 1, 6, 0.25, true, 0.01f0, 1.0f-3),
    )

    # head parameters
    ch_head_in = compute_channels(blocks[end].channels[2], params)
    ch_head_out = compute_channels(1280, params)

    # layers
    return Chain(
        # Stem
        Chain(
            Conv((3, 3), channels => ch_out; stride=2, bias, pad),
            BatchNorm(ch_out, swish),
        ),

        # MBConv blocks
        Chain(
            MBConvBlock(blocks[1], params)...,
            MBConvBlock(blocks[2], params)...,
            MBConvBlock(blocks[3], params)...,
            MBConvBlock(blocks[4], params)...,
            MBConvBlock(blocks[5], params)...,
            MBConvBlock(blocks[6], params)...,
            MBConvBlock(blocks[7], params)...,
        ),

        # Head
        !head ? identity : Chain(
            Conv((1, 1), ch_head_in => ch_head_out; bias, pad),
            BatchNorm(ch_head_out, swish),
            AdaptiveMeanPool((1, 1)),
        ),

        # Top
        !top ? identity : Chain(
            Flux.flatten,
            Dense(ch_head_out, classes),
        ),
    )
end

function extract_features(m::EfficientNet, x)
    ids = stages(m)
    layers = (
        m.layers[1],
        m.layers[2][1:ids[1]],
        m.layers[2][(ids[1] + 1):ids[2]],
        m.layers[2][(ids[2] + 1):ids[3]],
        m.layers[2][(ids[3] + 1):ids[4]],
    )
    return Flux.extraChain(layers, x)
end
