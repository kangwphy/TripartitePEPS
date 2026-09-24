#= Solve and save four directed boundaries of the fixed exact Q tensor.

Usage: sbatch jobs/run_cpu.sh bin/run_boundary.jl EVEN ODD OUTPUT_DIRECTORY
Output must be new. Checkpoints retain TensorKit spaces and parity sectors;
they are Julia Serialization files for this pinned package environment only.
=#
include("../src/FermionicPEPS.jl")
using .FermionicPEPS, TensorKit, PEPSKit, MPSKit
using Serialization, TOML, SHA, Dates

function main(args)
    length(args)==3 || error("usage: run_boundary.jl EVEN ODD OUTPUT_DIRECTORY")
    chi=(parse(Int,args[1]),parse(Int,args[2]))
    all(>=(0),chi) && sum(chi)>0 || error("invalid parity dimensions")
    output=abspath(args[3])
    ispath(output) && error("output already exists: $output")
    mkpath(output)
    root=normpath(joinpath(@__DIR__,".."))
    source_hashes=Dict(name=>bytes2hex(sha256(read(joinpath(root,"src",name))))
        for name in sort(readdir(joinpath(root,"src"))) if endswith(name,".jl"))
    provenance=Dict("created_utc"=>string(now(UTC)),"julia_version"=>string(VERSION),
        "model"=>"KSVC Q projector", "convention"=>"projector",
        "physical_tensor_optimized"=>false,"chi_even"=>chi[1],"chi_odd"=>chi[2],
        "tensor_leg_order"=>["physical","north","east","south","west"],
        "source_sha256"=>source_hashes,
        "manifest_sha256"=>bytes2hex(sha256(read(joinpath(root,"environments","default","Manifest.toml")))),
        "package_versions"=>Dict("TensorKit"=>string(Base.pkgversion(TensorKit)),
             "PEPSKit"=>string(Base.pkgversion(PEPSKit)),"MPSKit"=>string(Base.pkgversion(MPSKit))),
        "status"=>"running")
    open(joinpath(output,"run.toml"),"w") do io; TOML.print(io,provenance); end
    for direction in 1:4
        boundary=solve_boundary(;chi,direction,tolerance=1e-8)
        objects=lmps_objects(boundary)
        density=boundary_number(boundary)
        metadata=Dict("direction"=>direction,"converged"=>boundary.converged,
            "lambda_real"=>real(boundary.lambda),"lambda_imag"=>imag(boundary.lambda),
            "density_real"=>real(density),"density_imag"=>imag(density),
            "galerkin"=>boundary.galerkin,"iterations"=>boundary.iterations,
            "left_residual"=>boundary.left_residual,"right_residual"=>boundary.right_residual,
            "center_residual"=>boundary.center_residual,
            "lmps_residuals"=>Dict(string(k)=>v for (k,v) in pairs(objects.residuals)))
        temporary=joinpath(output,"direction_$direction.jls.partial")
        serialize(temporary,(;boundary,objects,metadata))
        mv(temporary,joinpath(output,"direction_$direction.jls"))
        open(joinpath(output,"direction_$direction.toml"),"w") do io; TOML.print(io,metadata); end
        @show direction boundary.lambda density objects.residuals
        flush(stdout)
    end
    provenance["status"]="complete_single_copy_boundaries"
    open(joinpath(output,"run.toml"),"w") do io; TOML.print(io,provenance); end
end

main(ARGS)
