using LMPSVUMPS, Serialization, SHA
let p=open(deserialize, ENV["TFIM_SELECTED_STATE"])
    bytes2hex(open(sha256, ENV["TFIM_SELECTED_STATE"])) == ENV["TFIM_BARE_SOURCE_SHA"] || error("GS hash mismatch")
    gs=p.selected_groundstate
    gs.converged || error("unconverged source")
    p.config.D == parse(Int, ENV["TFIM_BARE_D"]) || error("source D mismatch")
    p.config.h == parse(Float64, ENV["TFIM_BARE_H"]) || error("source h mismatch")
    p.config.ctm_chi == parse(Int, ENV["TFIM_CTM_CHI"]) || error("source CTM chi mismatch")
    p.summary_row.projected_gradient_norm <= parse(Float64, ENV["TFIM_AD_TOLERANCE"]) || error("source gradient failed")
end
include("solve_oriented_bare_cardinal.jl")
