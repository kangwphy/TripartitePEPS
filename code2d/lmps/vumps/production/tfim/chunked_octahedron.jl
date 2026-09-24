# Isolated exact contraction-order alternative; no production method override.
# Sum every value of one shared bond instead of storing rank-six intermediates.
function chunked_octahedron(Rs, vertices; hinges=Dict{String,Matrix{Float64}}())
    M = LMPSVUMPS
    out = M.m2_shared_tensors(Rs, vertices; hinges)
    li = [M.m2_vertex_index(vertices, name) for name in (:AB12, :AC13, :BC14)]
    ri = [M.m2_vertex_index(vertices, name) for name in (:AB34, :AC24, :BC23)]
    unmatched(a, b) = symdiff(Set(inds(a)), Set(inds(b)))
    # Pick a bridge already present in both first pair products. Slicing it
    # reduces both pair and triangle intermediates, not only the final dot.
    bridges = intersect(unmatched(out[li[1]], out[li[2]]),
                        unmatched(out[ri[1]], out[ri[2]]))
    isempty(bridges) && error("no bridge common to the two first pair products")
    edge = first(sort!(collect(bridges); by=M.m2_edge_key))
    owners = findall(R -> hasind(R, edge), out)
    length(owners) == 2 || error("sliced edge must have exactly two owners")
    z = zero(promote_type(map(eltype, out)...))
    peak = 0
    for value in 1:dim(edge)
        sliced = copy(out)
        selector = onehot(edge => value)
        for owner in owners
            sliced[owner] = sliced[owner] * selector
        end
        left = sliced[li[1]] * sliced[li[2]]
        peak = max(peak, prod(dim.(inds(left))))
        left *= sliced[li[3]]
        peak = max(peak, prod(dim.(inds(left))))
        right = sliced[ri[1]] * sliced[ri[2]]
        peak = max(peak, prod(dim.(inds(right))))
        right *= sliced[ri[3]]
        peak = max(peak, prod(dim.(inds(right))))
        result = left * right
        order(result) == 0 || error("sliced closure has open indices")
        z += scalar(result)
    end
    (; z, peak, sliced_edge=M.m2_edge_key(edge), slices=dim(edge))
end
