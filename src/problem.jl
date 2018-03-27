
mutable struct Problem

    alloc::Bool
    ptr::Ptr{Void}
    functions::Array{Function,1}
    constraints::Array{Constraint,1}
    net::Network

    function Problem(a, ptr, f, c, n)
        this = new(a, ptr, f, c, n)
        finalizer(this, dealloc)
        this
    end
end

Problem(net::Network) = Problem(true,
                                ccall((:PROB_new, libpfnet), Ptr{Void}, (Ptr{Void},), net.ptr),
                                Function[],
                                Constraint[],
                                net)

function add_constraint(prob::Problem, constr::Constraint)
    constr.alloc = false
    push!(prob.constraints, constr)
    ccall((:PROB_add_constr, libpfnet), Void, (Ptr{Void}, Ptr{Void},), prob.ptr, constr.ptr)
end

function add_function(prob::Problem, func::Function)
    func.alloc = false
    push!(prob.functions, func)
    ccall((:PROB_add_func, libpfnet), Void, (Ptr{Void}, Ptr{Void},), prob.ptr, func.ptr)
end
    
analyze(prob::Problem) = ccall((:PROB_analyze, libpfnet), Void, (Ptr{Void},), prob.ptr)

function dealloc(prob::Problem)
    if prob.alloc
        ccall((:PROB_del, libpfnet), Void, (Ptr{Void},), prob.ptr)
    end
    prob.alloc = false
    prob.ptr = C_NULL
end

b(prob::Problem) = Vector(ccall((:PROB_get_b, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)
f(prob::Problem) = Vector(ccall((:PROB_get_f, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)
l(prob::Problem) = Vector(ccall((:PROB_get_l, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)
u(prob::Problem) = Vector(ccall((:PROB_get_u, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)

G(prob::Problem) = SparseMatrixCSC(ccall((:PROB_get_G, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr))

show_problem(prob::Problem) = ccall((:PROB_show, libpfnet), Void, (Ptr{Void},), prob.ptr)
