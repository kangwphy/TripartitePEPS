using Test

# Isolate historical standalone fixtures so their module/global definitions
# cannot leak between cases. No external checkpoint is required by these cases.
cases = if isempty(ARGS)
    ["bell_pair_occupation", "replica_probe", "gauge_probe"]
elseif ARGS == ["all"]
    sort([splitext(f)[1] for f in readdir(joinpath(@__DIR__, "regression"))
          if endswith(f, ".jl")])
else
    copy(ARGS)
end
for name in cases
    occursin(r"^[A-Za-z0-9_]+$", name) || error("Invalid case name: $name")
    path = joinpath(@__DIR__, "regression", name * ".jl")
    isfile(path) || error("Unknown regression case: $name")
    isolated = gensym(:FermionRegression)
    @testset "$name" begin
        @eval module $isolated
            include($path)
        end
    end
end
