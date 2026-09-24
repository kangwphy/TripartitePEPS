using Test
using Random
using LinearAlgebra
using LMPSVUMPS
using TensorKit

@testset "LMPS VUMPS adapter keeps the RK tensor convention" begin
    source = rk_ising(0.30)
    peps = pepskit_from_lmps(source)
    raw = convert(Array, peps[1,1])
    @test size(raw) == (source.d, source.D, source.D, source.D, source.D)
    @test raw ≈ permutedims(source.A, (1,2,5,4,3)) atol=1e-14
    @test TensorKit.isdual(TensorKit.space(peps[1,1], 2)) # north
    @test TensorKit.isdual(TensorKit.space(peps[1,1], 3)) # east
    @test !TensorKit.isdual(TensorKit.space(peps[1,1], 4)) # south
    @test !TensorKit.isdual(TensorKit.space(peps[1,1], 5)) # west
    @test TensorKit.space(peps[1,1], 2) == dual(TensorKit.space(peps[1,1], 4))
    @test TensorKit.space(peps[1,1], 3) == dual(TensorKit.space(peps[1,1], 5))
end

@testset "quantum TFIM construction does not assume lattice symmetry" begin
    model = TransverseIsing2D(; J=1.0, h=0.7, unitcell=(2, 2))
    H = tfim_hamiltonian(model)
    @test size(H.lattice) == (2, 2)
    @test length(H.terms) > 0

    peps = tfim_initial_peps(model; D=2, seed=17)
    @test size(peps) == (2, 2)
    north = LMPSVUMPS._quantum_side_source(peps, :north)
    south = LMPSVUMPS._quantum_side_source(peps, :south)
    east = LMPSVUMPS._quantum_side_source(peps, :east)
    west = LMPSVUMPS._quantum_side_source(peps, :west)
    # The random unit cell is deliberately not C4 symmetric.  Each direction
    # must therefore retain a distinct rotated local tensor and be solved
    # independently by solve_quantum_vumps_boundaries.
    @test north[1, 1] == peps[1, 1]
    @test !isapprox(convert(Array, north[1, 1]), convert(Array, south[1, 1]); atol=1e-12)
    @test !isapprox(convert(Array, east[1, 1]), convert(Array, west[1, 1]); atol=1e-12)
    @test_throws ArgumentError TransverseIsing2D(; J=0.0)
end

@testset "local LTR index maps keep ket, bra, and retained bonds distinct" begin
    sentinel(offset) = reshape(
        ComplexF64.(offset .+ (1:16)) .+ im .* ComplexF64.(16:-1:1), 2, 2, 2, 2
    )
    GL, GR = sentinel(40), sentinel(60)
    C = ComplexF64[1.0+0.1im 0.2-0.3im; -0.1+0.2im 0.8-0.1im]
    L = LMPSVUMPS._left_environment_map(GL)
    R = LMPSVUMPS._right_environment_map(GR)
    @test size(L) == (2, 8)
    @test size(R) == (8, 2)
    for a in 1:2, l in 1:2, ket in 1:2, bra in 1:2
        x = ket + 2 * (bra - 1)
        @test L[a, l + 2 * (x - 1)] == GL[a, ket, bra, l]
    end
    for r in 1:2, ket in 1:2, bra in 1:2, a in 1:2
        x = ket + 2 * (bra - 1)
        @test R[r + 2 * (x - 1), a] == GR[r, ket, bra, a]
    end
    Rcenter = LMPSVUMPS._center_dress_right(R, C, 4)
    expected = zeros(ComplexF64, 8, 2)
    for oldleft in 1:2, x in 1:4, newright in 1:2, oldright in 1:2
        expected[oldleft + 2 * (x - 1), newright] +=
            C[oldleft, oldright] * R[oldright + 2 * (x - 1), newright]
    end
    @test Rcenter ≈ expected atol=1e-14
end
