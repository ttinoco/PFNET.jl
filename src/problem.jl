
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

function apply_heuristics(prob::Problem, var_values::Array{Float64,1})
    vec::Ptr{Void} = pfnet_vec_from_Array(var_values)
    ccall((:PROB_apply_heuristics, libpfnet), Void, (Ptr{Void}, Ptr{Void},), prob.ptr, vec)
    Base.Libc.free(vec)
end

function combine_H(prob::Problem, coeff::Array{Float64,1})
    vec::Ptr{Void} = pfnet_vec_from_Array(coeff)
    ccall((:PROB_combine_H, libpfnet), Void, (Ptr{Void}, Ptr{Void}, Bool,), prob.ptr, vec, false)
    Base.Libc.free(vec)
end

function dealloc(prob::Problem)
    if prob.alloc
        ccall((:PROB_del, libpfnet), Void, (Ptr{Void},), prob.ptr)
    end
    prob.alloc = false
    prob.ptr = C_NULL
end

function eval(prob::Problem, var_values::Array{Float64,1})
    vec::Ptr{Void} = pfnet_vec_from_Array(var_values)
    ccall((:PROB_eval, libpfnet), Void, (Ptr{Void}, Ptr{Void},), prob.ptr, vec)
    Base.Libc.free(vec)
end

network(prob::Problem) = Network(ccall((:PROB_get_network, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr))

num_primal_variables(prob::Problem) = ccall((:PROB_get_num_primal_variables, libpfnet), Int, (Ptr{Void},), prob.ptr)
num_linear_equality_constraints(prob::Problem) = ccall((:PROB_get_num_linear_equality_constraints, libpfnet), Int, (Ptr{Void},), prob.ptr)
num_nonlinear_equality_constraints(prob::Problem) = ccall((:PROB_get_num_nonlinear_equality_constraints, libpfnet), Int, (Ptr{Void},), prob.ptr)

x(prob::Problem) = Vector(ccall((:PROB_get_init_point, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), true) # julia owns memory

phi(prob::Problem) = ccall((:PROB_get_phi, libpfnet), Float64, (Ptr{Void},), prob.ptr)
gphi(prob::Problem) = Vector(ccall((:PROB_get_gphi, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)
Hphi(prob::Problem) = SparseMatrixCSC(ccall((:PROB_get_Hphi, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)

b(prob::Problem) = Vector(ccall((:PROB_get_b, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)
A(prob::Problem) = SparseMatrixCSC(ccall((:PROB_get_A, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)

f(prob::Problem) = Vector(ccall((:PROB_get_f, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)
J(prob::Problem) = SparseMatrixCSC(ccall((:PROB_get_J, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)

H_combined(prob::Problem) = SparseMatrixCSC(ccall((:PROB_get_H_combined, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr))

l(prob::Problem) = Vector(ccall((:PROB_get_l, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)
u(prob::Problem) = Vector(ccall((:PROB_get_u, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)
G(prob::Problem) = SparseMatrixCSC(ccall((:PROB_get_G, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), false)

show_problem(prob::Problem) = ccall((:PROB_show, libpfnet), Void, (Ptr{Void},), prob.ptr)
