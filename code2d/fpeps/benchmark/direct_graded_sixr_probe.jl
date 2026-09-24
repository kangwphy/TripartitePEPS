# Diagnostic only: no physical-accuracy claim follows from channel convergence.
include("direct_graded_lmps_probe.jl")
using KrylovKit

function direct_rails(boundary)
    x = direct_probe(boundary)
    println("direct channel direction=", boundary.direction, " ", x.diagnostics)
    flush(stdout)
    x.diagnostics["tail_converged"] || error("direct tails did not converge")
    x.diagnostics["bg_residual"] <= 1e-7 || error("direct generalized equation failed")
    singular = reduce(vcat, (svdvals(b) for (_, b) in blocks(x.G)))
    minimum(singular) > 1e-12maximum(singular) || error("unresolved direct support")
    H = x.B * inv(x.G)
    channel = y -> MPSKit.transfer_right(y, H, H)
    vals, vecs, info = eigsolve(channel, id(ComplexF64, space(H, 1)), 1, :LM;
                              tol=1e-12, maxiter=1000)
    info.converged >= 1 || error("LMPS self-transfer did not converge")
    radius = abs(vals[1])
    H /= sqrt(radius)
    # Undo the recorded auxiliary fusion, preserving its graded chart. The
    # physical ket/bra ports remain separate for occupation-replica sewing.
    objects = (; AL=H, L=x.L*x.UL, R=x.UR'*x.R, G=x.G,
               direction=boundary.direction)
    (; objects, rails=fermionic_rails(objects), direct=x, radius)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    input, output = ARGS
    mkpath(output)
    north = deserialize(joinpath(input, "boundary_1.jls"))
    south = deserialize(joinpath(input, "boundary_3.jls"))
    a, c = direct_rails(north), direct_rails(south)
    seams = regional_seam_rails((; A=a.rails, B=a.rails, C=c.rails))
    cap_result = solve_seam_caps(seams)
    serialize(joinpath(output, "prepared.jls"), (; a, c, seams, cap_result))
    result = measure_lmps_stilde(seams, cap_result.caps; depths=(8,16,32,64),
        accept_unconverged=true, on_sample=s -> begin
            println("depth=", s.depth, " stilde=", s.entropies.stilde,
                    " endpoint_residual=", s.endpoint_residual)
            flush(stdout)
        end)
    serialize(joinpath(output, "result.jls"), result)
    open(joinpath(output, "result.toml"), "w") do io
        TOML.print(io, Dict("stilde"=>result.stilde,
            "length_converged"=>result.converged,
            "physical_replica_benchmark_certified"=>false,
            "construction"=>"graded direct grown tails, auxiliary fusion undone on rails"))
    end
end
