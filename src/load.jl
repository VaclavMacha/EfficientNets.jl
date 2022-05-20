function _load!(layer::Conv; weight, bias = nothing)
    weight_perm = PermutedDimsArray(weight, (4, 3, 2, 1))
    for inds in eachindex(weight_perm)
        i, j, k, m = Tuple(inds)
        layer.weight[end-i+1, end-j+1, k, m] = weight_perm[i, j, k, m]
    end
    if !isnothing(bias)
        copyto!(layer.bias, bias)
    end
    return
end

function _load!(layer::BatchNorm; γ, β, μ, σ²)
    copyto!(layer.γ, γ)
    copyto!(layer.β, β)
    copyto!(layer.μ, μ)
    copyto!(layer.σ², σ²)
    return
end

function _load!(layer::Dense; weight, bias)
    copyto!(layer.weight, weight)
    copyto!(layer.bias, bias)
    return
end

function _load!(model::MBConv, pars, base)
    layers = isa(model.layers, SkipConnection) ? model.layers.layers : model.layers

    # Expansion phase
    expansion = layers[1]
    if !isa(expansion, typeof(identity))
        _load!(expansion[1]; weight = pars[base*"._expand_conv.weight"])
        _load!(expansion[2];
            γ=pars[base*"._bn0.weight"],
            β=pars[base*"._bn0.bias"],
            μ=pars[base*"._bn0.running_mean"],
            σ²=pars[base*"._bn0.running_var"],
        )
    end

    # Depthwise convolution phase
    depthwise = layers[2]
    _load!(depthwise[1]; weight = pars[base*"._depthwise_conv.weight"])
    _load!(depthwise[2];
        γ=pars[base*"._bn1.weight"],
        β=pars[base*"._bn1.bias"],
        μ=pars[base*"._bn1.running_mean"],
        σ²=pars[base*"._bn1.running_var"]
    )

    # Squeeze and Excitation layer
    squeeze = isa(layers[3], SkipConnection) ? layers[3].layers : layers[3]
    if !isa(squeeze, typeof(identity))
        _load!(squeeze[2];
            weight=pars[base*"._se_reduce.weight"],
            bias=pars[base*"._se_reduce.bias"],
        )
        _load!(squeeze[3];
            weight=pars[base*"._se_expand.weight"],
            bias=pars[base*"._se_expand.bias"]
        )
    end

    # Pointwise convolution phase
    point = layers[4]
    _load!(point[1]; weight = pars[base*"._project_conv.weight"])
    _load!(point[2];
        γ=pars[base*"._bn2.weight"],
        β=pars[base*"._bn2.bias"],
        μ=pars[base*"._bn2.running_mean"],
        σ²=pars[base*"._bn2.running_var"],
    )
    return
end

function _load!(model::EfficientNet{M}, path; load_top::Bool = true) where {M}
    pars = Pickle.Torch.THload(path)
    layers = model.layers

    # Stem
    stem = layers[1]
    _load!(stem[1]; weight=pars["_conv_stem.weight"])
    _load!(stem[2];
        γ=pars["_bn0.weight"],
        β=pars["_bn0.bias"],
        μ=pars["_bn0.running_mean"],
        σ²=pars["_bn0.running_var"]
    )

    # MBConv blocks
    for (i, block) in enumerate(layers[2])
        _load!(block, pars, "_blocks.$(i - 1)")
    end

    # Head
    head = layers[3]
    if !isa(head, typeof(identity))
        _load!(head[1]; weight=pars["_conv_head.weight"])
        _load!(head[2];
            γ=pars["_bn1.weight"],
            β=pars["_bn1.bias"],
            μ=pars["_bn1.running_mean"],
            σ²=pars["_bn1.running_var"]
        )
    end

    # Top
    top = layers[4]
    if !isa(top, typeof(identity)) && load_top
        _load!(top[2];
            weight=pars["_fc.weight"],
            bias=pars["_fc.bias"]
        )
    end
    return
end
