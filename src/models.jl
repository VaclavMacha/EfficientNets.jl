abstract type ModelName end

name(T::Type{<:ModelName}) = string(T.name.name)

"""
    parameters(::Type{<:ModelName})

Specifies parameters of the model based on the model name. The function returns a tuple of `(width, depth, resolution, dropout)`.

"""
function parameters() end

url(::Type{<:ModelName}) = nothing
urlhash(::Type{<:ModelName}) = nothing

url_adv(::Type{<:ModelName}) = nothing
urlhash_adv(::Type{<:ModelName}) = nothing

function load_path(T::Type{<:ModelName}; adversarial::Bool=false)
    file = adversarial ? url_adv(T) : url(T)
    if isnothing(file)
        type = adversarial ? "adversarial" : ""
        msg = "Pre-trained $(type) model is not available for EfficientNet-$(T)"
        throw(ArgumentError(msg))
    end
    modelpath = "EfficientNet-$(name(T))/$(basename(file))"
    return @datadep_str modelpath
end

# Predefined models
const URL_BASE = "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0"

struct B0 <: ModelName end

parameters(::Type{B0}) = (1.0, 1.0, 224, 0.2)
stages(::Type{B0}) = (3, 5, 9, 16)
stages_channels(::Type{B0}) = (32, 24, 40, 112, 320)

url(::Type{B0}) = joinpath(URL_BASE, "efficientnet-b0-355c32eb.pth")
urlhash(::Type{B0}) = "355c32ebf5c7b2e19ed8f926daea5b6d71c8744a49aab3148fc90775fbaac5a3"

url_adv(::Type{B0}) = joinpath(URL_BASE, "adv-efficientnet-b0-b64d5a18.pth")
urlhash_adv(::Type{B0}) = "b64d5a18952a215048ee8541f46f0d0fb253eee749f780d22c6aa57561de5c3e"

struct B1 <: ModelName end

parameters(::Type{B1}) = (1.0, 1.1, 240, 0.2)
stages(::Type{B1}) = (5, 8, 16, 23)
stages_channels(::Type{B1}) = (32, 24, 40, 112, 320)

url(::Type{B1}) = joinpath(URL_BASE, "efficientnet-b1-f1951068.pth")
urlhash(::Type{B1}) = "f19510681ee65135bad30718410bff1f9c51452b858a6db2f61b7707e18b933d"

url_adv(::Type{B1}) = joinpath(URL_BASE, "adv-efficientnet-b1-0f3ce85a.pth")
urlhash_adv(::Type{B1}) = "0f3ce85a0ecbc16831f7f9e6202f35dd34a6642ba571984bf5281b6aa860d47a"

struct B2 <: ModelName end

parameters(::Type{B2}) = (1.1, 1.2, 260, 0.3)
stages(::Type{B2}) = (5, 8, 16, 23)
stages_channels(::Type{B2}) = (32, 24, 48, 120, 352)

url(::Type{B2}) = joinpath(URL_BASE, "efficientnet-b2-8bb594d6.pth")
urlhash(::Type{B2}) = "8bb594d63dcd2315ffd3fe5b6dc4380c3cc8ce4209bdf63a5d76963ce36845c0"

url_adv(::Type{B2}) = joinpath(URL_BASE, "adv-efficientnet-b2-6e9d97e5.pth")
urlhash_adv(::Type{B2}) = "6e9d97e5ee02bdfd722a128c4db18110999316af5e40a6f30b074ad0db1057db"

struct B3 <: ModelName end

parameters(::Type{B3}) = (1.2, 1.4, 300, 0.3)
stages(::Type{B3}) = (5, 8, 18, 26)
stages_channels(::Type{B3}) = (40, 32, 48, 136, 384)

url(::Type{B3}) = joinpath(URL_BASE, "efficientnet-b3-5fb5a3c3.pth")
urlhash(::Type{B3}) = "5fb5a3c3f1d45062d3d3dbe770f1a7daf78eed140f1c092cceef9ab79677d08e"

url_adv(::Type{B3}) = joinpath(URL_BASE, "adv-efficientnet-b3-cdd7c0f4.pth")
urlhash_adv(::Type{B3}) = "cdd7c0f4d1e8d97387d00bc8d0ccfe55364c232e9cb04ae6f1b00d20912feb33"

struct B4 <: ModelName end

parameters(::Type{B4}) = (1.4, 1.8, 380, 0.4)
stages(::Type{B4}) = (6, 10, 22, 32)
stages_channels(::Type{B4}) = (48, 32, 56, 160, 448)

url(::Type{B4}) = joinpath(URL_BASE, "efficientnet-b4-6ed6700e.pth")
urlhash(::Type{B4}) = "6ed6700e361e3878dc53f285f3df4abf9f39f6da23545cac065b7dc9467a8959"

url_adv(::Type{B4}) = joinpath(URL_BASE, "adv-efficientnet-b4-44fb3a87.pth")
urlhash_adv(::Type{B4}) = "44fb3a87ef763da422a700d9d1f62328263ce3d86074e92766d28e3e1ceb2408"

struct B5 <: ModelName end

parameters(::Type{B5}) = (1.6, 2.2, 456, 0.4)
stages(::Type{B5}) = (8, 13, 27, 39)
stages_channels(::Type{B5}) = (48, 40, 64, 176, 512)

url(::Type{B5}) = joinpath(URL_BASE, "efficientnet-b5-b6417697.pth")
urlhash(::Type{B5}) = "b641769706a245cea68d309c3d35a864b6317e686e315fe79ed3d81e78c6f51c"

url_adv(::Type{B5}) = joinpath(URL_BASE, "adv-efficientnet-b5-86493f6b.pth")
urlhash_adv(::Type{B5}) = "86493f6be52121a1908efaa228e1ea396d28d5155468b4cfbc145d8f71cede2c"

struct B6 <: ModelName end

parameters(::Type{B6}) = (1.8, 2.6, 528, 0.5)
stages(::Type{B6}) = (9, 15, 31, 45)
stages_channels(::Type{B6}) = (56, 40, 72, 200, 576)

url(::Type{B6}) = joinpath(URL_BASE, "efficientnet-b6-c76e70fd.pth")
urlhash(::Type{B6}) = "c76e70fd23f007dd6b09c8464e4e184a81d4d3d715ee42c2c89650c0be701505"

url_adv(::Type{B6}) = joinpath(URL_BASE, "adv-efficientnet-b6-ac80338e.pth")
urlhash_adv(::Type{B6}) = "ac80338ed82705fa2710279c583263071ccd249ab856f682f1be52d6da14ff4e"

struct B7 <: ModelName end

parameters(::Type{B7}) = (2.0, 3.1, 600, 0.5)
stages(::Type{B7}) = (11, 18, 38, 55)
stages_channels(::Type{B7}) = (64, 48, 80, 224, 640)

url(::Type{B7}) = joinpath(URL_BASE, "efficientnet-b7-dcc49843.pth")
urlhash(::Type{B7}) = "dcc49843b8ec5a097a8cb166eded1d7a09291093eff7735f603bcfcafce12c8e"

url_adv(::Type{B7}) = joinpath(URL_BASE, "adv-efficientnet-b7-4652b6dd.pth")
urlhash_adv(::Type{B7}) = "4652b6dd826ae720eb0ad60b03a6893dd18c18b5527781660292bd9dd75e9ccd"

struct B8 <: ModelName end

parameters(::Type{B8}) = (2.2, 3.6, 672, 0.5)

url_adv(::Type{B8}) = joinpath(URL_BASE, "adv-efficientnet-b8-22a8fe65.pth")
urlhash_adv(::Type{B8}) = "22a8fe65d42c4dc0599dfdf0519f823f7eb301ae65aa20fa32b32319767e93c8"

struct L2 <: ModelName end

parameters(::Type{L2}) = (4.3, 5.3, 800, 0.5)
