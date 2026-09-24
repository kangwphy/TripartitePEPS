# Recover xi_M from the exact original boundary checkpoint used for bare S.
using LMPSVUMPS, Serialization, SHA, LinearAlgebra, DelimitedFiles
include("bare_strip_checks.jl")
BLAS.set_num_threads(parse(Int,get(ENV,"SLURM_CPUS_PER_TASK","1")))
const PACKAGE=abspath(joinpath(@__DIR__,"..",".."))
const DATA=joinpath(PACKAGE,"data","quantum_tfim")
const CURRENT=joinpath(DATA,"current")
const ARCHIVE=joinpath(CURRENT,"_archive_legacy")
sha(path)=bytes2hex(open(sha256,path))
function saved_path(path,expected; fallback=nothing)
    candidates=[path]
    startswith(path,CURRENT*"/") && push!(candidates,replace(path,CURRENT*"/"=>ARCHIVE*"/";count=1))
    if startswith(path,DATA*"/") && !startswith(path,CURRENT*"/")
        for kind in ("gs_runs","measurements","stilde","gs")
            push!(candidates,replace(path,DATA*"/"=>joinpath(ARCHIVE,kind)*"/";count=1))
        end
    end
    fallback===nothing || push!(candidates,fallback)
    matches=unique(realpath(p) for p in candidates if isfile(p) && sha(p)==expected)
    isempty(matches) && error("missing checkpoint with expected SHA: $path")
    first(matches)
end
input,output=ARGS
raw,header=readdlm(input,',',String;header=true)
keys=vec(header)
field(row,key)=row[findfirst(==(key),keys)]
cols=["D","h","chi","source_sha256","tildeS","xi_H","xi_M","cardinal_input","cardinal_sha256","checkpoint","checkpoint_sha256"]
tmp=output*".tmp."*get(ENV,"SLURM_JOB_ID","manual")
open(tmp,"w") do io
    println(io,join(cols,','));flush(io)
    for row in eachrow(raw)
        cp=field(row,"checkpoint");expected=field(row,"source_sha256")
        sha(cp)==field(row,"checkpoint_sha256") || error("S checkpoint changed")
        sha(field(row,"source_state"))==expected || error("GS checkpoint changed")
        p=open(deserialize,cp);r=p.row
        r.source_state_sha256==expected || error("GS mismatch within S checkpoint")
        isapprox(r.tildeS,parse(Float64,field(row,"tildeS"));atol=1e-13,rtol=1e-12) || error("S mismatch")
        cardinal=saved_path(r.cardinal_input,r.cardinal_input_sha256;fallback=joinpath(dirname(dirname(cp)),"closure.jls"))
        c=open(deserialize,cardinal)
        c.row.passed && c.row.source_state_sha256==expected || error("invalid closure")
        all(b.converged for b in c.boundaries) || error("boundary not converged")
        M=c.M.north
        size(M,1)==parse(Int,field(row,"chi")) || error("chi mismatch")
        ratios=BareStripChecks.spectrum(M)
        length(ratios)>1 && 0<ratios[2]<1 || error("invalid original boundary spectrum")
        xi=-inv(log(ratios[2]))
        values=[field(row,"D"),field(row,"h"),field(row,"chi"),expected,field(row,"tildeS"),field(row,"xi_H"),string(xi),cardinal,r.cardinal_input_sha256,cp,field(row,"checkpoint_sha256")]
        println(io,join(values,','));flush(io)
        println("MEASURED h=",field(row,"h")," chi=",field(row,"chi")," xi_M=",xi);flush(stdout)
    end
end
mv(tmp,output;force=true)
