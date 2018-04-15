module pfnet

# Dependencies
if isfile(joinpath(dirname(@__FILE__),"..","deps","deps.jl"))
    include("../deps/deps.jl")
else
    error("PFNET is not properly installed. Please run Pkg.build(\"pfnet\")")
end

# Vector
function Vector(ptr::Ptr{Void}, own::Bool)
    return unsafe_wrap(Array,
                       ccall((:VEC_get_data, libpfnet), Ptr{Float64}, (Ptr{Void},), ptr),
                       ccall((:VEC_get_size, libpfnet), Int, (Ptr{Void},), ptr),
                       own)
end

function pfnet_vec_from_Array{T}(ar::Array{T,1})
    return ccall((:VEC_new_from_array, libpfnet),
                 Ptr{Void},
                 (Ptr{T}, Int,),
                 Ref(ar, 1),
                 size(ar)[1])
end

# SparseMatrixCSC
function SparseMatrixCSC(ptr::Ptr{Void})
    m = ccall((:MAT_get_size1, libpfnet), Int, (Ptr{Void},), ptr)
    n = ccall((:MAT_get_size2, libpfnet), Int, (Ptr{Void},), ptr)
    nnz = m = ccall((:MAT_get_nnz, libpfnet), Int, (Ptr{Void},), ptr)
    I = unsafe_wrap(Array,
                    ccall((:MAT_get_row_array, libpfnet), Ptr{Int32}, (Ptr{Void},), ptr),
                    nnz,
                    false)+1
    J = unsafe_wrap(Array,
                    ccall((:MAT_get_col_array, libpfnet), Ptr{Int32}, (Ptr{Void},), ptr),
                    nnz,
                    false)+1
    V = unsafe_wrap(Array,
                    ccall((:MAT_get_data_array, libpfnet), Ptr{Float64}, (Ptr{Void},), ptr),
                    nnz,
                    false)
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
