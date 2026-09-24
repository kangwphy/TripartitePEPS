# sbatch jobs/run_cpu.sh bin/run_finite_stilde.jl L TOTAL_CHI NEW_OUTPUT
include("../src/FermionicPEPS.jl")
using .FermionicPEPS, TensorKit, Serialization, TOML, SHA, Dates

function main_finite(args)
    length(args) in 3:5 || error("usage: run_finite_stilde.jl L TOTAL_CHI NEW_OUTPUT [WIDTH [CUT]] (CHI=0 disables truncation)")
    L,rank=parse.(Int,args[1:2]); chi=rank==0 ? nothing : rank
    iseven(L) && L>=2 && rank>=0 || error("even L>=2 and CHI>=0 required")
    width=length(args)>=4 ? parse(Int,args[4]) : L
    cut=length(args)>=5 ? parse(Int,args[5]) : width÷2
    width>=2 && 0<cut<width || error("width>=2 and 0<cut<width required")
    output=abspath(args[3]); ispath(output) && error("output already exists: $output")
    weights=parse.(Float64,split(get(ENV,"FPEPS_BOND_WEIGHT","0.125,0.125"),','))
    length(weights)==1 && (weights=repeat(weights,2))
    length(weights)==2 && all(isfinite,weights) || error("one or two finite real bond weights required")
    gauge=get(ENV,"FPEPS_GAUGE","balanced")
    gauge in ("outgoing","balanced") || error("FPEPS_GAUGE must be outgoing or balanced")
    peps=ksvc_ipeps(;bond_weight=Tuple(weights),bond_gauge=Symbol(gauge))
    rotate=get(ENV,"FPEPS_ROTATE_180","false")=="true"
    rotate && (peps=rot180(peps))
    phase_tolerance=parse(Float64,get(ENV,"FPEPS_PHASE_TOLERANCE","1e-8"))
    root=normpath(joinpath(@__DIR__,".."))
    mkpath(output)
    metadata=Dict{String,Any}("L"=>L,"width"=>width,"cut"=>cut,"chi"=>rank,"bond_weight"=>weights,"gauge"=>gauge,
        "rotate_180"=>rotate,"physical_D"=>2,"method"=>"finite_region_mps",
        "regulator"=>"virtual_vacuum_obc","junctions"=>1,"convention"=>"occupation_ABC",
        "created_utc"=>string(now(UTC)),"julia_version"=>string(VERSION),
        "phase_tolerance"=>phase_tolerance,"size_converged"=>false,"chi_converged"=>false,
        "source_sha256"=>Dict(f=>bytes2hex(sha256(read(joinpath(root,"src",f))))
            for f in readdir(joinpath(root,"src")) if endswith(f,".jl")),
        "manifest_sha256"=>bytes2hex(sha256(read(joinpath(root,"environments","default","Manifest.toml")))),
        "status"=>"running")
    function save_metadata()
        path=joinpath(output,"run.toml")
        open(path*".partial","w") do io; TOML.print(io,metadata); end
        mv(path*".partial",path;force=true)
    end
    save_metadata()
    serialize(joinpath(output,"peps.jls"),peps)
    open(joinpath(output,"sectors_scaled.csv"),"w") do io
        println(io,"sector,real,imag,logscale,cancellation_condition")
    end
    try
        run=measure_finite_fpeps_stilde(peps;L,width,cut,chi,phase_tolerance,
            on_region=(name,m)->begin
                serialize(joinpath(output,"region_$name.jls"),m)
                metadata["region_$name"]=Dict("row_discarded"=>m.discarded,"row_ranks"=>m.ranks,"logscale"=>m.logscale)
                save_metadata()
                @show name m.ranks m.discarded
                flush(stdout)
            end,
            on_sector=(name,z)->begin
                serialize(joinpath(output,"$name.jls"),z)
                open(joinpath(output,"sectors_scaled.csv"),"a") do io
                    println(io,join((name,real(z.value),imag(z.value),z.logscale,z.cancellation_condition),','))
                end
                @show name z.value z.logscale z.cancellation_condition
                flush(stdout)
            end)
        serialize(joinpath(output,"result.jls"),run)
        open(joinpath(output,"sectors_scaled.csv"),"w") do io
            println(io,"sector,real,imag,logscale,cancellation_condition")
            for (name,z) in pairs(run.sectors)
                println(io,join((name,real(z.value),imag(z.value),z.logscale,z.cancellation_condition),','))
            end
        end
        open(joinpath(output,"stilde.csv"),"w") do io; println(io,run.stilde); end
        metadata["status"]="finite_window_measured"
        metadata["stilde"]=run.stilde
        metadata["S2"]=run.entropies.S2
        metadata["S3"]=run.entropies.S3
        metadata["diagnostics"]=Dict(string(k)=>v for (k,v) in pairs(run.diagnostics))
        @show run.stilde run.diagnostics
    catch err
        metadata["status"]="failed"
        metadata["error"]=sprint(showerror,err)
        rethrow()
    finally
        metadata["finished_utc"]=string(now(UTC))
        save_metadata()
    end
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    main_finite(ARGS)
end
