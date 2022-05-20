module EfficientNets

using Flux

using Flux: @functor
using Flux: flatten
using Flux: SkipConnection

export EfficientNet
export B0, B1, B2, B3, B4, B5, B6, B7, B8, L2

export extract_features
export stages
export stages_channels

# Custom types
struct ModelParams{T<:Real,I<:Integer}
    image_size::NTuple{2,I}
    width_coef::T
    depth_coef::T
    n_classes::I
    depth_divisor::I
    depth_min::I
    dropout::T
    include_top::Bool
end

struct BlockParams{T<:Real,I<:Integer}
    repeats::I
    filters::NTuple{2,I}
    channels::Pair{I,I}
    stride::I
    expand_ratio::I
    squeeze_ratio::T
    skip_connection::Bool
    momentum::T
    ϵ::T
end

include("models.jl")
include("mbconv.jl")
include("efficientnet.jl")

end # module
