abstract type ModelName end

Base.string(T::Type{<:ModelName}) = string(T.name.name)

"""
    parameters(::Type{<:ModelName})

Specifies parameters of the model based on the model name. The function returns a tuple of `(width, depth, resolution, dropout)`.

"""
function parameters() end

struct B0 <: ModelName end

parameters(::Type{B0}) = (1.0, 1.0, 224, 0.2)

struct B1 <: ModelName end

parameters(::Type{B1}) = (1.0, 1.1, 240, 0.2)

struct B2 <: ModelName end

parameters(::Type{B2}) = (1.1, 1.2, 260, 0.3)


struct B3 <: ModelName end

parameters(::Type{B3}) = (1.2, 1.4, 300, 0.3)

struct B4 <: ModelName end

parameters(::Type{B4}) = (1.4, 1.8, 380, 0.4)

struct B5 <: ModelName end

parameters(::Type{B5}) = (1.6, 2.2, 456, 0.4)

struct B6 <: ModelName end

parameters(::Type{B6}) = (1.8, 2.6, 528, 0.5)

struct B7 <: ModelName end

parameters(::Type{B7}) = (2.0, 3.1, 600, 0.5)

struct B8 <: ModelName end

parameters(::Type{B8}) = (2.2, 3.6, 672, 0.5)

struct L2 <: ModelName end

parameters(::Type{L2}) = (4.3, 5.3, 800, 0.5)
