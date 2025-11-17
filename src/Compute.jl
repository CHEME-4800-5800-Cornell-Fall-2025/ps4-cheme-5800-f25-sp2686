function _safe_log(x::Float64)
    if x <= 0.0
        return -1.0e10
    else
        return log(x)
    end
end

function _objective_function(w::Vector{Float64}, ḡ::Vector{Float64},
    Σ̂::Matrix{Float64}, R::Float64, μ::Float64, ρ::Float64; include_barrier::Bool = true)

    # variance term
    quad = dot(w, Σ̂ * w)

    # equality-constraint penalty (stronger as ρ -> 0)
    penalty = (1.0 / (2.0 * ρ)) * ((sum(w) - 1.0)^2 + (dot(ḡ, w) - R)^2)

    if include_barrier
        barrier = - (1.0 / μ) * sum(_safe_log.(w))
        return quad + penalty + barrier
    else
        return quad + penalty
    end
end

# simple projection to non-negative simplex
function project_to_simplex(v::Vector{Float64})
    v .= max.(v, 0.0)
    s = sum(v)
    if s == 0.0
        return fill(1.0/length(v), length(v))
    else
        return v ./ s
    end
end

function solve(model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem;
    verbose::Bool = true, K::Int = 10000, T₀::Float64 = 1.0, T₁::Float64 = 1e-6,
    α::Float64 = 0.99, β::Float64 = 0.01, τ::Float64 = 0.95,
    μ::Float64 = 1.0, ρ::Float64 = 1.0; include_barrier::Bool = true)

    # unpack (work on copies)
    ḡ = copy(model.ḡ)
    Σ̂ = copy(model.Σ̂)
    R = model.R
    d = length(model.w)

    # normalize initial w to simplex
    current_w = project_to_simplex(copy(model.w))

    # initial objective
    current_f = _objective_function(current_w, ḡ, Σ̂, R, μ, ρ; include_barrier=include_barrier)
    w_best = copy(current_w)
    f_best = current_f

    T = T₀
    KL = max(1, K)
    min_eps = 1e-12

    while T > T₁
        accepted_counter = 0
        for k in 1:KL
            candidate = current_w .+ β .* randn(d)
            # enforce non-negativity and simplex
            candidate .= max.(candidate, min_eps)
            candidate = candidate ./ sum(candidate)

            candidate_f = _objective_function(candidate, ḡ, Σ̂, R, μ, ρ; include_barrier=include_barrier)

            if candidate_f <= current_f || rand() < exp(-(candidate_f - current_f) / max(T, 1e-16))
                current_w = candidate
                current_f = candidate_f
                accepted_counter += 1
                if current_f < f_best
                    w_best = copy(current_w)
                    f_best = current_f
                end
            end
        end

        # adapt KL modestly
        fraction_accepted = accepted_counter / max(KL,1)
        if fraction_accepted > 0.8
            KL = max(10, ceil(Int, 0.75*KL))
        elseif fraction_accepted < 0.2
            KL = ceil(Int, 1.5*KL)
        end

        # decrease penalty parameters (so 1/μ and 1/ρ grow)
        μ *= τ
        ρ *= τ

        # cooling
        T *= α

        if verbose
            @info "SA" T=T KL=KL fraction_accepted=fraction_accepted f_best=f_best
        end
    end

    # final attempt: enforce equalities exactly via small QP projection (if JuMP+MadNLP available)
    proj_w = nothing
    try
        using JuMP, MadNLP
        qp = Model(()->MadNLP.Optimizer(print_level=MadNLP.ERROR, max_iter=500))
        @variable(qp, w[1:d] >= 0.0, start = w_best)
        @objective(qp, Min, sum((w[i]-w_best[i])^2 for i in 1:d))
        @constraint(qp, sum(w) == 1.0)
        @constraint(qp, dot(ḡ, w) == R)
        optimize!(qp)
        if is_solved_and_feasible(qp)
            proj_w = value.(w)
        end
    catch
        proj_w = nothing
    end

    if proj_w === nothing
        # fallback heuristic: nudge along ḡ while preserving sum==1
        w_tmp = copy(w_best)
        for i in 1:50
            curr_r = dot(ḡ, w_tmp)
            err = R - curr_r
            if abs(err) < 1e-6
                break
            end
            v = ḡ .- mean(ḡ) # sum(v)==0
            denom = dot(ḡ, v)
            if abs(denom) < 1e-14
                break
            end
            λ = err / denom
            w_tmp .= w_tmp .+ λ .* v
            w_tmp .= max.(w_tmp, 0.0)
            if sum(w_tmp) == 0.0
                w_tmp .= 1.0/d
            else
                w_tmp ./= sum(w_tmp)
            end
        end
        proj_w = w_tmp
    end

    proj_w .= max.(proj_w, 0.0)
    proj_w ./= sum(proj_w)

    model.w = copy(proj_w)
    return model
end

function solve(problem::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem)::Dict{String,Any}
    results = Dict{String,Any}()
    Σ = problem.Σ
    μvec = problem.μ
    R = problem.R
    bounds = problem.bounds
    wₒ = problem.initial

    d = length(μvec)
    model = Model(()->MadNLP.Optimizer(print_level=MadNLP.ERROR, max_iter=500))
    @variable(model, bounds[i,1] <= w[i=1:d] <= bounds[i,2], start = wₒ[i])

    @objective(model, Min, transpose(w)*Σ*w)

    @constraints(model, begin
        transpose(μvec)*w == R
        sum(w) == 1.0
    end)

    optimize!(model)

    @assert is_solved_and_feasible(model)

    w_opt = value.(w)
    results["argmax"] = w_opt
    results["reward"] = transpose(μvec)*w_opt
    results["objective_value"] = objective_value(model)
    results["status"] = termination_status(model)

    return results
end