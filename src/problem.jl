importall MathProgBase.SolverInterface

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

function apply_heuristics(prob::Problem, var_values::Base.Vector{Float64})
    vec::Ptr{Void} = pfnet_vec_from_Array(var_values)
    ccall((:PROB_apply_heuristics, libpfnet), Void, (Ptr{Void}, Ptr{Void},), prob.ptr, vec)
    Base.Libc.free(vec)
end

function combine_H(prob::Problem, coeff::Base.Vector{Float64})
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

function eval(prob::Problem, var_values::Base.Vector{Float64})
    vec::Ptr{Void} = pfnet_vec_from_Array(var_values)
    ccall((:PROB_eval, libpfnet), Void, (Ptr{Void}, Ptr{Void},), prob.ptr, vec)
    Base.Libc.free(vec)
end

network(prob::Problem) = Network(ccall((:PROB_get_network, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr))

num_primal_variables(prob::Problem) = ccall((:PROB_get_num_primal_variables, libpfnet), Int, (Ptr{Void},), prob.ptr)
num_linear_equality_constraints(prob::Problem) = ccall((:PROB_get_num_linear_equality_constraints, libpfnet), Int, (Ptr{Void},), prob.ptr)
num_nonlinear_equality_constraints(prob::Problem) = ccall((:PROB_get_num_nonlinear_equality_constraints, libpfnet), Int, (Ptr{Void},), prob.ptr)

init_point(prob::Problem) = Vector(ccall((:PROB_get_init_point, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), true) # julia owns memory
upper_limits(prob::Problem) = Vector(ccall((:PROB_get_upper_limits, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), true) # julia owns memory
lower_limits(prob::Problem) = Vector(ccall((:PROB_get_lower_limits, libpfnet), Ptr{Void}, (Ptr{Void},), prob.ptr), true) # julia owns memory
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

# min phi
# Ax = b
# f(x) = 0
# l <= Gx <= u
function solve(prob::Problem, solver::AbstractMathProgSolver)

    ma = size(A(prob))[1]
    mf = size(f(prob))[1]
    mg = size(G(prob))[1]
    nx = length(x(prob))
    ns = mg
    
    # min phi
    # 0 <= Ax-b <= 0
    # 0 <= f(x) - 0 <= 0
    # 0 <= Gx - s <= 0
    # [-inf l] <= [x s] <= [inf u]

    numVar = nx + ns
    numConstr = ma+mf+mg
    l = [lower_limits(prob); pfnet.l(prob)]
    u = [upper_limits(prob); pfnet.u(prob)]
    lb = zeros(ma+mf+mg)
    ub = zeros(ma+mf+mg)
    sense = :Min
    model = NonlinearModel(solver)
    evaluator = ProblemEvaluator(prob, nx, ns, ma, mf, mg)
    y = [x(prob); rand(ns)]
    
    # Testing
    """
    println(nx)
    println(ns)
    initialize(evaluator, [:Jac])
    println(features_available(evaluator))
    println(eval_f(evaluator, y))
    g = zeros(ma+mf+mg)
    eval_g(evaluator, g, y)
    g = zeros(nx+ns)
    eval_grad_f(evaluator, g, y)
    Ig, Jg = jac_structure(evaluator)
    Ih, Jh = hesslag_structure(evaluator)
    Jg = zeros(length(Ig))
    Vg = eval_jac_g(evaluator, Jg, y)
    Hh = zeros(length(Ih))
    Vh = eval_hesslag(evaluator, Hh, y, 0.3, rand(numConstr))
    """

    loadproblem!(model, numVar, numConstr, l, u, lb, ub, sense, evaluator)
    setwarmstart!(model, y)
    optimize!(model)    
end

mutable struct ProblemEvaluator <: AbstractNLPEvaluator
    problem::Problem
    nx::Int
    ns::Int
    ma::Int
    mf::Int
    mg::Int
end

function initialize(d::ProblemEvaluator, requested_features::Base.Vector{Symbol})
    analyze(d.problem)
end

function features_available(d::ProblemEvaluator)
    return Base.Vector([:Grad, :Jac, :Hess])
end

function eval_f(d::ProblemEvaluator, y)
    eval(d.problem, y[1:d.nx])
    return phi(d.problem)
end

function eval_g(d::ProblemEvaluator, g, y)
    x = y[1:d.nx]
    s = y[d.nx+1:end]
    eval(d.problem, x)
    g[:] = [A(d.problem)*x-b(d.problem);
            f(d.problem);
            G(d.problem)*x-s]            
end

function eval_grad_f(d::ProblemEvaluator, g, y)
    eval(d.problem, y[1:d.nx])
    g[:] = [gphi(d.problem); zeros(d.ns)]
end

function jac_structure(d::ProblemEvaluator)
    Arow, Acol, Aval = SparseMatrixCOO(ccall((:PROB_get_A, libpfnet), Ptr{Void}, (Ptr{Void},), d.problem.ptr), false)
    Jrow, Jcol, Jval = SparseMatrixCOO(ccall((:PROB_get_J, libpfnet), Ptr{Void}, (Ptr{Void},), d.problem.ptr), false)
    Grow, Gcol, Gval = SparseMatrixCOO(ccall((:PROB_get_G, libpfnet), Ptr{Void}, (Ptr{Void},), d.problem.ptr), false)
    return ([Arow; Jrow+d.ma; Grow+d.ma+d.mf; (d.ma+d.mf+1):(d.ma+d.mf+d.mg)],
            [Acol; Jcol; Gcol; (d.nx+1):(d.nx+d.ns)])
end

function hesslag_structure(d::ProblemEvaluator)
    H1row, H1col, H1val = SparseMatrixCOO(ccall((:PROB_get_Hphi, libpfnet), Ptr{Void}, (Ptr{Void},), d.problem.ptr), false)
    H2row, H2col, H2val = SparseMatrixCOO(ccall((:PROB_get_H_combined, libpfnet), Ptr{Void}, (Ptr{Void},), d.problem.ptr), false)
    return([H1row; H2row],
           [H1col; H2col])
end

function eval_jac_g(d::ProblemEvaluator, J, y)
    x = y[1:d.nx]
    s = y[d.nx+1:end]
    eval(d.problem, x)
    Arow, Acol, Aval = SparseMatrixCOO(ccall((:PROB_get_A, libpfnet), Ptr{Void}, (Ptr{Void},), d.problem.ptr), false)
    Jrow, Jcol, Jval = SparseMatrixCOO(ccall((:PROB_get_J, libpfnet), Ptr{Void}, (Ptr{Void},), d.problem.ptr), false)
    Grow, Gcol, Gval = SparseMatrixCOO(ccall((:PROB_get_G, libpfnet), Ptr{Void}, (Ptr{Void},), d.problem.ptr), false)
    J[:] = [Aval; Jval; Gval; -ones(length(s))]
end

function eval_hesslag(d::ProblemEvaluator, H, y, sigma, mu)
    eval(d.problem, y[1:d.nx])
    combine_H(d.problem, mu[d.ma+1:d.ma+d.mf])
    H1row, H1col, H1val = SparseMatrixCOO(ccall((:PROB_get_Hphi, libpfnet), Ptr{Void}, (Ptr{Void},), d.problem.ptr), false)
    H2row, H2col, H2val = SparseMatrixCOO(ccall((:PROB_get_H_combined, libpfnet), Ptr{Void}, (Ptr{Void},), d.problem.ptr), false)
    H[:] = [sigma*H1val; H2val]
end
