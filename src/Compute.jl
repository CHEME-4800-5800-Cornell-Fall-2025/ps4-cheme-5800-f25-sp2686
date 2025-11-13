function _safe_log(x::Float64)
    if x <= 0.0
        return -1.0e10  # a large negative number
    else
        return log(x)
    end
end

function _objective_function(w::Vector{Float64}, ḡ::Vector{Float64},
    Σ̂::Matrix{Float64}, R::Float64, μ::Float64, ρ::Float64; include_barrier::Bool = false)

    # quadratic variance term (scalar)
    quad = dot(w, Σ̂ * w)

    # penalty for equality constraints (scalar)
    penalty = (1.0 / (2.0 * ρ)) * ((sum(w) - 1.0)^2 + (dot(ḡ, w) - R)^2)

    if include_barrier
        barrier = - (1.0 / μ) * sum(_safe_log.(w))
        return quad + penalty + barrier
    else
        return quad + penalty
    end
end

function solve(model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem;
    verbose::Bool = true, K::Int = 10000, T₀::Float64 = 1.0, T₁::Float64 = 0.1,
    α::Float64 = 0.99, β::Float64 = 0.01, τ::Float64 = 0.99,
    μ::Float64 = 1.0, ρ::Float64 = 1.0; include_barrier::Bool = false)

    # unpack model parameters (use copies to avoid mutating input)
    w = copy(model.w)
    ḡ = model.ḡ
    Σ̂ = model.Σ̂
    R = model.R

    # simulated annealing params
    T = T₀
    current_w = copy(w)
    current_f = _objective_function(current_w, ḡ, Σ̂, R, μ, ρ; include_barrier=include_barrier)

    w_best = copy(current_w)
    f_best = current_f
    KL = max(1, K)

    d = length(w)
    min_eps = 1e-12
    has_converged = false

    while !has_converged
        accepted_counter = 0

        # inner Metropolis loop
        for k in 1:KL
            candidate = current_w .+ β .* randn(d)
            # enforce small positive floor (required if using barrier)
            candidate = max.(candidate, min_eps)

            f_candidate = _objective_function(candidate, ḡ, Σ̂, R, μ, ρ; include_barrier=include_barrier)

            accept = false
            if f_candidate <= current_f
                accept = true
            else
                Δ = f_candidate - current_f
                accept_prob = exp(-Δ / max(T, 1e-16))
                accept = rand() < accept_prob
            end

            if accept
                current_w = candidate
                current_f = f_candidate
                accepted_counter += 1

                if current_f < f_best
                    w_best = copy(current_w)
                    f_best = current_f
                end
            end
        end

        # adapt KL based on acceptance rate
        fraction_accepted = accepted_counter / max(KL, 1)
        if fraction_accepted > 0.8
            KL = max(10, ceil(Int, 0.75 * KL))
        elseif fraction_accepted < 0.2
            KL = ceil(Int, 1.5 * KL)
        end

        # update penalty parameters (scale by τ)
        μ *= τ
        ρ *= τ

        # cooling
        if T <= T₁
            has_converged = true
        else
            T *= α
        end

        if verbose
            @info "SA step" T=T KL=KL fraction_accepted=fraction_accepted f_best=f_best
        end
    end

    model.w = w_best
    return model
end

function solve(problem::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem)::Dict{String,Any}
    results = Dict{String,Any}()
    Σ = problem.Σ
    μ = problem.μ
    R = problem.R
    bounds = problem.bounds
    wₒ = problem.initial

    d = length(μ)
    model = Model(()->MadNLP.Optimizer(print_level=MadNLP.ERROR, max_iter=500))
    @variable(model, bounds[i,1] <= w[i=1:d] <= bounds[i,2], start=wₒ[i])

    @objective(model, Min, transpose(w)*Σ*w)

    @constraints(model,
        begin
            transpose(μ)*w >= R
            sum(w) == 1.0
        end
    )

    optimize!(model)

    @assert is_solved_and_feasible(model)

    w_opt = value.(w)
    results["argmax"] = w_opt
    results["reward"] = transpose(μ)*w_opt
    results["objective_value"] = objective_value(model)
    results["status"] = termination_status(model)

    return results
end