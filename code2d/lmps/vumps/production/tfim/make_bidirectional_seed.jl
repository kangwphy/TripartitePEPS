# Fresh product-core D3 seeds. No checkpoint is read or deserialized.
# Even this small numerical construction runs only in a Slurm allocation.
using LMPSVUMPS, PEPSKit, TensorKit
using LinearAlgebra, Serialization, SHA, Printf

seed_file_sha(path) = bytes2hex(open(SHA.sha256, path))
seed_json_string(value) = "\"" * replace(string(value), '\\' => "\\\\", '"' => "\\\"",
    '\n' => "\\n", '\r' => "\\r", '\t' => "\\t") * "\""
seed_json(value::Union{AbstractString,Symbol}) = seed_json_string(value)
seed_json(value::Bool) = value ? "true" : "false"
seed_json(value::Real) = isfinite(value) ? string(value) : error("Nonfinite seed metadata")
seed_json(value::AbstractVector) = "[" * join(seed_json.(value), ',') * "]"
seed_json(value::NamedTuple) = "{" * join((seed_json_string(k) * ":" * seed_json(v) for (k,v) in pairs(value)), ',') * "}"
function seed_atomic_write(writer, path)
    temporary = path * ".tmp." * ENV["SLURM_JOB_ID"]
    try
        open(writer, temporary, "w")
        isfile(path) && error("Refusing to replace existing fresh seed artifact: $path")
        mv(temporary, path; force=false)
    finally
        isfile(temporary) && rm(temporary)
    end
end

# The metadata is produced by this generator, with a flat scalar recipe.
# Validate unique exact fields without loading the serialized PEPS.
function seed_json_field(text, key, pattern)
    matches = collect(eachmatch(Regex("\\\"" * key * "\\\"\\s*:\\s*" * pattern), text))
    length(matches) == 1 || error("Fresh seed metadata must contain one $key")
    return matches[1].captures[1]
end
seed_json_text(text, key) = seed_json_field(text, key, "\\\"([^\\\"]*)\\\"")
seed_json_number(text, key) = parse(Float64, seed_json_field(text, key, "([-+0-9.eE]+)"))
function validate_existing_seed(output, recipe, tensor_sha)
    metadata = output * ".json"
    isfile(output) && isfile(metadata) || error("Incomplete seed pair; review it rather than overwriting")
    text = read(metadata, String)
    for key in (:type, :source, :campaign, :branch, :product_core, :script_sha256)
        seed_json_text(text, string(key)) == string(getproperty(recipe, key)) || error("Existing seed recipe differs: $key")
    end
    for key in (:D, :h, :epsilon, :seed)
        seed_json_number(text, string(key)) == Float64(getproperty(recipe, key)) || error("Existing seed recipe differs: $key")
    end
    seed_json_field(text, "uses_old_checkpoint", "(true|false)") == "false" || error("Seed claims an old checkpoint")
    seed_json_text(text, "peps_tensor_sha256") == tensor_sha || error("Existing seed tensor recipe is not reproduced")
    expected = seed_json_text(text, "checkpoint_sha256")
    occursin(r"^[0-9a-f]{64}$", expected) && seed_file_sha(output) == expected || error("Existing seed file SHA mismatch")
    println("FRESH_SEED_REUSED checkpoint=$output sha256=$expected (metadata/hash only; no deserialize)")
    return expected
end

function make_bidirectional_seed(options)
    isempty(get(ENV, "SLURM_JOB_ID", "")) && error("Generate seeds in a Slurm allocation, not on a login node")
    branch = options["branch"]
    branch in ("ordered", "disordered") || error("Branch must be ordered or disordered")
    D = parse(Int, get(options, "D", "3"))
    D == 3 || error("This fresh campaign fixes D=3")
    default_h, default_seed, core = branch == "ordered" ?
        (3.038, 2026100601, "up_z") : (3.056, 2026100602, "plus_x")
    h = parse(Float64, get(options, "h", string(default_h)))
    seed = parse(Int, get(options, "seed", string(default_seed)))
    epsilon = parse(Float64, get(options, "epsilon", "0.05"))
    isfinite(h) && h == default_h || error("Fresh branch endpoint must be h=$default_h")
    seed == default_seed || error("Fresh branch seed must be $default_seed")
    epsilon == 0.05 || error("This controlled campaign fixes relative perturbation at 0.05")
    output = abspath(options["output"])
    # Protect existing history: outputs must belong to the new fresh campaign.
    "qr_bidirectional_fresh_20261006" in splitpath(output) || error("Output must be inside qr_bidirectional_fresh_20261006")
    endswith(output, ".jls") || error("Output must have .jls extension")
    script_sha = seed_file_sha(@__FILE__)
    recipe = (; type="product_plus_c4v_noise", source="generated_from_scratch",
        campaign="qr_bidirectional_fresh_20261006", uses_old_checkpoint=false,
        D, h, branch, epsilon, seed, product_core=core, scalar_type="ComplexF64",
        unitcell="1x1", virtual_core="only_north_east_south_west_1111",
        physical_basis="Z=diag(1,-1);X=[0,1;1,0]",
        noise_recipe="complex_Gaussian_C4v_then_remove_real_product_projection_then_C4v_again_then_scale_relative_norm",
        physical_Z2_constraint=false, script_sha256=script_sha,
        canonical_code_revision=get(ENV, "TFIM_CODE_REVISION", "1e7cfd3"))

    # The canonical random initializer supplies spaces and deterministic full-D
    # Gaussian noise only. It is never a historical optimized state.
    carrier = tfim_initial_peps(TransverseIsing2D(; J=1.0, h, unitcell=(1,1));
        D, seed, scalar_type=ComplexF64)
    raw_product = zeros(ComplexF64, 2, D, D, D, D)
    spin = branch == "ordered" ? ComplexF64[1,0] : ComplexF64[1,1] / sqrt(2)
    raw_product[:,1,1,1,1] .= spin
    product = PEPSKit.InfinitePEPS(TensorKit.TensorMap(raw_product, space(carrier[1,1])))
    product_norm = norm(product[1,1])
    product_norm > 0 || error("Zero product tensor")
    noise = PEPSKit.symmetrize!(deepcopy(carrier), PEPSKit.RotateReflect())[1,1]
    noise_before_projection = norm(noise)
    # RotateReflect includes conjugation, so its fixed manifold is real-linear.
    # Subtract a real product coefficient and reproject before a real scaling.
    projection = real(dot(product[1,1], noise)) / real(dot(product[1,1], product[1,1]))
    noise = noise - projection * product[1,1]
    noise = PEPSKit.symmetrize!(PEPSKit.InfinitePEPS(noise), PEPSKit.RotateReflect())[1,1]
    orthogonal_noise_norm = norm(noise)
    isfinite(orthogonal_noise_norm) && orthogonal_noise_norm > eps(Float64) || error("Degenerate C4v perturbation")
    noise = (epsilon * product_norm / orthogonal_noise_norm) * noise
    actual_noise_relative = norm(noise) / product_norm
    product_noise_overlap = abs(dot(product[1,1], noise)) / (product_norm * norm(noise))
    peps = PEPSKit.peps_normalize(PEPSKit.InfinitePEPS(product[1,1] + noise))
    symmetrized = PEPSKit.symmetrize!(deepcopy(peps), PEPSKit.RotateReflect())
    symmetry_error = norm(symmetrized[1,1] - peps[1,1]) / norm(peps[1,1])
    abs(actual_noise_relative - epsilon) <= 1e-14 && product_noise_overlap <= 1e-12 && symmetry_error <= 1e-12 || error("Seed recipe checks failed")
    dense = Array(peps[1,1])
    size(dense) == (2,D,D,D,D) && eltype(dense) == ComplexF64 || error("Unexpected dense seed tensor layout")
    singular_min, singular_max, virtual_condition = Float64[], Float64[], Float64[]
    for virtual_axis in 2:5
        axes_order = [virtual_axis; [i for i in 1:5 if i != virtual_axis]]
        singular = svdvals(reshape(permutedims(dense, axes_order), D, :))
        minimum(singular) > 0 || error("Rank-deficient virtual seed leg $virtual_axis")
        push!(singular_min, minimum(singular));push!(singular_max, maximum(singular))
        push!(virtual_condition, maximum(singular) / minimum(singular))
    end
    tensor_sha = bytes2hex(SHA.sha256(reinterpret(UInt8, vec(dense))))
    core_mx, core_mz = branch == "ordered" ? (0.0,1.0) : (1.0,0.0)
    information = (; recipe..., product_tensor_norm=product_norm,
        C4v_noise_norm_before_projection=noise_before_projection,
        C4v_noise_norm_after_projection_before_scale=orthogonal_noise_norm,
        actual_noise_relative_norm=actual_noise_relative, product_noise_overlap,
        normalized_seed_tensor_norm=norm(peps[1,1]), C4v_relative_error=symmetry_error,
        virtual_axes="north,east,south,west", virtual_min_singular_values=singular_min,
        virtual_max_singular_values=singular_max, virtual_condition_numbers=virtual_condition,
        peps_tensor_sha256=tensor_sha, analytic_product_core_mx=core_mx,
        analytic_product_core_mz=core_mz, magnetization_scope="analytic_unperturbed_product_core_only",
        actual_perturbed_PEPS_magnetization_measured=false)
    if isfile(output) || isfile(output * ".json")
        return validate_existing_seed(output, recipe, tensor_sha)
    end
    mkpath(dirname(output))
    seed_atomic_write(output) do io
        serialize(io, (; format_version=1, peps, D, h, config=recipe,
            initialization_info=information, uses_old_checkpoint=false,
            source="generated_from_scratch"))
    end
    checkpoint_sha = seed_file_sha(output)
    seed_atomic_write(output * ".json") do io
        println(io, seed_json((; information..., checkpoint=output, checkpoint_sha256=checkpoint_sha)))
    end
    println("FRESH_SEED_CREATED checkpoint=$output sha256=$checkpoint_sha tensor_sha256=$tensor_sha core=$core noise=$actual_noise_relative")
    return checkpoint_sha
end

function fresh_seed_options(arguments)
    iseven(length(arguments)) || error("Options require --name value pairs")
    options = Dict{String,String}()
    allowed = Set(["D", "branch", "output", "h", "seed", "epsilon"])
    for i in 1:2:length(arguments)
        startswith(arguments[i], "--") || error("Expected --option")
        name = arguments[i][3:end]
        name in allowed && !haskey(options, name) || error("Unknown or repeated --$name")
        options[name] = arguments[i+1]
    end
    haskey(options, "branch") && haskey(options, "output") || error("Usage: make_bidirectional_seed.jl --branch ordered|disordered --output FRESH_CAMPAIGN/seeds/NAME.jls")
    return options
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    make_bidirectional_seed(fresh_seed_options(ARGS))
end
