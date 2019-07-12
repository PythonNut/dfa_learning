module NaPS

using MathOptInterface
const MOI = MathOptInterface
const CI = MOI.ConstraintIndex
const VI = MOI.VariableIndex

const MOIU = MOI.Utilities

const PBTerm = MOI.ScalarAffineFunction{Int}

struct Solution
    ret_val::MOI.TerminationStatusCode
    primal::Vector{Int}
    objval::Int
end

Solution() = Solution(MOI.OPTIMIZE_NOT_CALLED, Float64[], 0)

const IntegerLinearTerm = Tuple{Vector{Int}, Vector{Int}}

# Used to build the data with allocate-load during `copy_to`.
# When `optimize!` is called, a the data is used to build `ECOSMatrix`
# and the `ModelData` struct is discarded
mutable struct ModelData
    m::Int # Number of rows/constraints
    n::Int # Number of cols/variables
    le::Vector{PBTerm}
    eq::Vector{PBTerm}
    objective::PBTerm
end

mutable struct ConstraintData
    l::Int
    e::Int
    z::Int

    function ConstraintData()
        new(0, 0, 0)
    end
end

mutable struct Optimizer <: MOI.AbstractOptimizer
    constraints::ConstraintData
    data::Union{Nothing, ModelData} # only non-Nothing between MOI.copy_to and MOI.optimize!
    sol::Solution
    options
    function Optimizer(; kwargs...)
        new(ConstraintData(), nothing, Solution(), kwargs)
    end
end

MOI.get(::Optimizer, ::MOI.SolverName) = "NaPS"

function MOI.is_empty(instance::Optimizer)
    instance.data === nothing
end

function MOI.empty!(instance::Optimizer)
    instance.data = nothing # It should already be nothing except if an error is thrown inside copy_to
    instance.sol = Solution()
end

MOIU.supports_allocate_load(::Optimizer, copy_names::Bool) = !copy_names

function MOI.supports(::Optimizer,
                      ::Union{MOI.ObjectiveSense,
                              MOI.ObjectiveFunction{MOI.SingleVariable},
                              MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}})
    return true
end

function MOI.supports_constraint(::Optimizer,
                                 ::Type{<:Union{MOI.SingleVariable}},
                                 ::Type{<:Union{MOI.ZeroOne}})
    return true
end

function MOI.supports_constraint(::Optimizer,
                                 ::Type{<:Union{MOI.ScalarAffineFunction{Float64}}},
                                 ::Type{<:Union{MOI.LessThan,
                                                MOI.EqualTo}})
    return true
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike; kws...)
    return MOIU.automatic_copy_to(dest, src; kws...)
end

constroffset(constraints::ConstraintData, ci::CI{<:MOI.AbstractFunction, MOI.ZeroOne}) = ci.value
function _allocate_constraint(constraints::ConstraintData, f::MOI.SingleVariable, s::MOI.ZeroOne)
    ci = constraints.z
    constraints.z += MOI.dimension(s)
    ci
end

constroffset(constraints::ConstraintData, ci::CI{<:MOI.AbstractFunction, MOI.LessThan}) = ci.value
function _allocate_constraint(constraints::ConstraintData, f::MOI.ScalarAffineFunction{Float64}, s::MOI.LessThan)
    ci = constraints.l
    constraints.l += MOI.dimension(s)
    ci
end

constroffset(constraints::ConstraintData, ci::CI{<:MOI.AbstractFunction, MOI.EqualTo}) = ci.value
function _allocate_constraint(constraints::ConstraintData, f::MOI.ScalarAffineFunction{Float64}, s::MOI.EqualTo)
    ci = constraints.e
    constraints.e += MOI.dimension(s)
    ci
end

constroffset(instance::Optimizer, ci::CI) = constroffset(instance.constraints, ci::CI)
function MOIU.allocate_constraint(instance::Optimizer, f::F, s::S) where {F <: MOI.AbstractFunction, S <: MOI.AbstractSet}
    CI{F, S}(_allocate_constraint(instance.constraints, f, s))
end

# Build constraint matrix
output_index(t::MOI.VectorAffineTerm) = t.output_index
variable_index_value(t::MOI.ScalarAffineTerm) = t.variable_index.value
variable_index_value(t::MOI.VectorAffineTerm) = variable_index_value(t.scalar_term)
coefficient(t::MOI.ScalarAffineTerm) = t.coefficient
coefficient(t::MOI.VectorAffineTerm) = coefficient(t.scalar_term)

set_value(s::MOI.LessThan) = s.upper
set_value(s::MOI.EqualTo) = s.value

function discretize_function(f::MOI.ScalarAffineFunction{Float64}, s=MOI.EqualTo(0)::Union{MOI.LessThan, MOI.EqualTo})
    return PBTerm([
        MOI.ScalarAffineTerm{Int}(Int(t.coefficient), t.variable_index) for t in f.terms
    ], Int(f.constant - set_value(s)))
end

function MOIU.load_constraint(instance::Optimizer, ci, f::MOI.SingleVariable, s::MOI.ZeroOne)
    # We get these for free!
end

function MOIU.load_constraint(instance::Optimizer, ci, f::MOI.ScalarAffineFunction{Float64}, s::MOI.LessThan)
    push!(instance.data.le, discretize_function(f, s))
end

function MOIU.load_constraint(instance::Optimizer, ci, f::MOI.ScalarAffineFunction{Float64}, s::MOI.EqualTo)
    push!(instance.data.eq, discretize_function(f, s))
end

function MOIU.allocate_variables(instance::Optimizer, nvars::Integer)
    instance.constraints = ConstraintData()
    VI.(1:nvars)
end

function MOIU.load_variables(instance::Optimizer, nvars::Integer)
    constraints = instance.constraints
    m = constraints.e + constraints.l
    instance.data = ModelData(m, nvars, [], [], PBTerm([], 0))
end

function MOIU.allocate(instance::Optimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    @assert sense == MOI.MIN_SENSE
end
function MOIU.allocate(::Optimizer, ::MOI.ObjectiveFunction,
                       ::MOI.Union{MOI.SingleVariable,
                                   MOI.ScalarAffineFunction{Float64}})
    # No need to allocate!
end

function MOIU.load(::Optimizer, ::MOI.ObjectiveSense, ::MOI.OptimizationSense)
end
function MOIU.load(optimizer::Optimizer, ::MOI.ObjectiveFunction,
                   f::MOI.SingleVariable)
    MOIU.load(optimizer,
              MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
              MOI.ScalarAffineFunction{Float64}(f))
end
function MOIU.load(instance::Optimizer, ::MOI.ObjectiveFunction,
                   f::MOI.ScalarAffineFunction)
    instance.data.objective = discretize_function(f)
    return nothing
end

function pb_format_coeff(c::Int)
    if c > 0
        return "+" * string(c)
    else
        return string(c)
    end
end

function pb_format_term(f::PBTerm)
    join((pb_format_coeff(coefficient(t)) * " x" * string(variable_index_value(t)) for t in f.terms), ' ')
end

function pb_format_eq(f::PBTerm)
    return pb_format_term(f) * " == $(-f.constant) ;"
end

function pb_format_le(f::PBTerm)
    return pb_format_term(f) * " <= $(-f.constant) ;"
end

function pb_format_objective(f::PBTerm)
    return "min: " * pb_format_term(f) * " ;"
end

function MOI.optimize!(instance::Optimizer)
    if instance.data === nothing
        # optimize! has already been called and no new model has been copied
        return
    end

    if instance.constraints.z != instance.data.n
        error("All variables must be binary to use NaPS")
    end


    mktemp() do path, io
        println(io, pb_format_objective(instance.data.objective))

        for f in instance.data.eq
            println(io, pb_format_eq(f))
        end

        for f in instance.data.le
            println(io, pb_format_le(f))
        end

        flush(io)

        NAPS_PATH = `./naps/naps $(path)`

        inp = Pipe()
        out = Pipe()
        err = Pipe()
        process = run(pipeline(NAPS_PATH, stdin=inp, stdout=out, stderr=err), wait=false)

        close(out.in)
        close(err.in)
        close(inp)

        # stdout = @async String(read(out))
        # stderr = @async String(read(err))

        # wait(process)

        assignments = []
        status = MOI.OPTIMIZE_NOT_CALLED
        optimum = 0

        while !eof(out)
            l = readline(out)
            if !(length(l) > 1 && l[1] == 'v')
                println(l)
            end

            if length(l) > 1
                if l[1] == 'v'
                    append!(assignments, split(l, ' ')[2:end])
                elseif l[1] == 's'
                    if l[3:end] in ("UNSUPPORTED", "UNKNOWN")
                        status = MOI.OTHER_ERROR
                    elseif l[3:end] == "SATISFIABLE"
                        status = MOI.LOCALLY_SOLVED
                        empty!(assignments)
                    elseif l[3:end] == "OPTIMUM FOUND"
                        status = MOI.OPTIMAL
                        empty!(assignments)
                    elseif l[3:end] == "UNSATISFIABLE"
                        status = MOI.INFEASIBLE
                    end
                elseif l[1] == 'o'
                    optimum = parse(Int, l[3:end])
                end
            end
        end

        flat = zeros(Int, instance.data.n)
        for ass in assignments
            if ass[1] == '-'
                key = parse(Int, ass[3:end])
                flat[key] = 0
            else
                key = parse(Int, ass[2:end])
                flat[key] = 1
            end
        end

        instance.sol = Solution(status, flat, optimum)
    end
end

# Implements getter for result value and statuses
function MOI.get(instance::Optimizer, ::MOI.TerminationStatus)
    return instance.sol.ret_val
end

function MOI.get(instance::Optimizer, ::MOI.PrimalStatus)
    flag = instance.sol.ret_val
    if flag == 0
        return MOI.FEASIBLE_POINT
    else
        return MOI.OTHER_RESULT_STATUS
    end
end

function MOI.get(instance::Optimizer, ::MOI.VariablePrimal, vi::VI)
    instance.sol.primal[vi.value]
end

MOI.get(instance::Optimizer, ::MOI.ObjectiveValue) = instance.sol.objval

MOI.get(instance::Optimizer, ::MOI.ResultCount) = 1

end
