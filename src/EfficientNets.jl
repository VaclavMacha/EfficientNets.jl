module EfficientNets

using Flux

using Flux: @functor
using Flux: flatten
using Flux: SkipConnection


include("models.jl")
include("mbconv.jl")

end # module
