module EfficientNets

using Flux

using Flux: @functor
using Flux: flatten
using Flux: SkipConnection

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

end # module
