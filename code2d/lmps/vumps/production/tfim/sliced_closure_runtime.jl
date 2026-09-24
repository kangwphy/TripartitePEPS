# Independent runtime entry: never overwrite a method used by live jobs.
using ITensors
import ITensors: dim, inds, hasind, onehot, scalar, order
include(joinpath(@__DIR__, "chunked_octahedron.jl"))

function sliced_octahedron_entry(Rs, vertices; kwargs...)
    result = chunked_octahedron(Rs, vertices; kwargs...)
    println("exact octahedron slicing: edge=$(result.sliced_edge) slices=$(result.slices) peak=$(result.peak)")
    flush(stdout)
    result.z, result.peak
end

# Copy the full current pipeline verbatim except its name and final
# octahedron contraction calls. All fixed-point, scale and closure gates stay.
let core = joinpath(@__DIR__, "../../src/algorithms/sixr_closure.jl")
    source = read(core, String)
    start = findfirst("function m2_contract_shared_six_R(", source)
    start === nothing && error("cannot locate full six-R pipeline")
    stop = findnext("\n\"\"\"", source, last(start))
    stop === nothing && error("cannot locate pipeline end")
    body = source[first(start):first(stop)-1]
    length(findall("m2_close_octahedron(", body)) == 2 || error("unexpected octahedron call sites")
    body = replace(body, "function m2_contract_shared_six_R(" =>
        "function m2_contract_shared_six_R_sliced(",
        "m2_close_octahedron(" => "Main.sliced_octahedron_entry(")
    Base.include_string(LMPSVUMPS, body, "isolated_exact_sliced_sixr_pipeline.jl")
end

function sliced_contract_regional_six_R(rails::NamedTuple, D::Int; kwargs...)
    all(hasproperty(rails, r) for r in (:A, :B, :C)) || error("three independent regional rails required")
    LMPSVUMPS.m2_contract_shared_six_R_sliced(rails, D; kwargs...)
end
