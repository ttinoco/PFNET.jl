module pfnet

# Dependencies
if isfile(joinpath(dirname(@__FILE__),"..","deps","deps.jl"))
    include("../deps/deps.jl")
else
    error("PFNET is not properly installed. Please run Pkg.build(\"pfnet\")")
end

# Vector
function Vector(vec::Ptr{Void}, own::Bool)
    jvec = unsafe_wrap(Array,
                       ccall((:VEC_get_data, libpfnet), Ptr{Float64}, (Ptr{Void},), vec),
                       ccall((:VEC_get_size, libpfnet), Int, (Ptr{Void},), vec),
                       own)
    if own
        Base.Libc.free(vec) # delete container
    end
    return jvec
end

function pfnet_vec_from_Array{T}(ar::Array{T,1})
    return ccall((:VEC_new_from_array, libpfnet),
                 Ptr{Void},
                 (Ptr{T}, Int,),
                 Ref(ar, 1),
                 size(ar)[1])
end

# Matrix
function SparseMatrixCOO(mat::Ptr{Void}, own::Bool)
    nnz = m = ccall((:MAT_get_nnz, libpfnet), Int, (Ptr{Void},), mat)
    I = unsafe_wrap(Array,
                    ccall((:MAT_get_row_array, libpfnet), Ptr{Int32}, (Ptr{Void},), mat),
                    nnz,
                    own)+1
    J = unsafe_wrap(Array,
                    ccall((:MAT_get_col_array, libpfnet), Ptr{Int32}, (Ptr{Void},), mat),
                    nnz,
                    own)+1
    V = unsafe_wrap(Array,
                    ccall((:MAT_get_data_array, libpfnet), Ptr{Float64}, (Ptr{Void},), mat),
                    nnz,
                    own)
    if own
        Base.Libc.free(mat) # delete container
    end
    return (I,J,V)
end

function SparseMatrixCSC(mat::Ptr{Void}, own::Bool)
    m = ccall((:MAT_get_size1, libpfnet), Int, (Ptr{Void},), mat)
    n = ccall((:MAT_get_size2, libpfnet), Int, (Ptr{Void},), mat)
    I, J, V = SparseMatrixCOO(mat, own)
    return sparse(I, J, V, m, n)
end

# Includes
include("strings.jl")
include("net.jl")
include("parser.jl")
include("function.jl")
include("constraint.jl")
include("problem.jl")

# Symbols
for name in names(pfnet, true)
    @eval export $(Symbol(name))
end

end
