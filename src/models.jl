abstract type ModelName end

Base.string(T::Type{<:ModelName}) = string(T.name.name)

"""
    parameters(::Type{<:ModelName})

Specifies parameters of the model based on the model name. The function returns a tuple of `(width, depth, resolution, dropout)`.

"""
function parameters() end

struct B0 <: ModelName end

parameters(::Type{B0}) = (1.0, 1.0, 224, 0.2)
stages(::Type{B0}) = (3, 5, 9, 16)
stages_channels(::Type{B0}) = (32, 24, 40, 112, 320)


struct B1 <: ModelName end

parameters(::Type{B1}) = (1.0, 1.1, 240, 0.2)
stages(::Type{B1}) = (5, 8, 16, 23)
stages_channels(::Type{B1}) = (32, 24, 40, 112, 320)


struct B2 <: ModelName end

parameters(::Type{B2}) = (1.1, 1.2, 260, 0.3)
stages(::Type{B2}) = (5, 8, 16, 23)
stages_channels(::Type{B2}) = (32, 24, 48, 120, 352)



struct B3 <: ModelName end

parameters(::Type{B3}) = (1.2, 1.4, 300, 0.3)
stages(::Type{B3}) = (5, 8, 18, 26)
stages_channels(::Type{B3}) = (40, 32, 48, 136, 384)


struct B4 <: ModelName end

parameters(::Type{B4}) = (1.4, 1.8, 380, 0.4)
stages(::Type{B4}) = (6, 10, 22, 32)
stages_channels(::Type{B4}) = (48, 32, 56, 160, 448)


struct B5 <: ModelName end

parameters(::Type{B5}) = (1.6, 2.2, 456, 0.4)
stages(::Type{B5}) = (8, 13, 27, 39)
stages_channels(::Type{B5}) = (48, 40, 64, 176, 512)


struct B6 <: ModelName end

parameters(::Type{B6}) = (1.8, 2.6, 528, 0.5)
stages(::Type{B6}) = (9, 15, 31, 45)
stages_channels(::Type{B6}) = (56, 40, 72, 200, 576)


struct B7 <: ModelName end

parameters(::Type{B7}) = (2.0, 3.1, 600, 0.5)
stages(::Type{B7}) = (11, 18, 38, 55)
stages_channels(::Type{B7}) = (64, 48, 80, 224, 640)


struct B8 <: ModelName end

parameters(::Type{B8}) = (2.2, 3.6, 672, 0.5)

struct L2 <: ModelName end

parameters(::Type{L2}) = (4.3, 5.3, 800, 0.5)
