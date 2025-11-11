function _safe_log(x::Float64)
    if x <= 0.0
        return -1.0e10; # a large negative number
    else
        return log(x);
    end
end

function _objective_function(w::Array{Float64,1}, ḡ::Array{Float64,1}, 
    Σ̂::Array{Float64,2}, R::Float64, μ::Float64, ρ::Float64)


    # TODO: This version of the objective function includes the barrier term, and the penalty terms -
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
    
    f = w'*(Σ̂*w) + (1/(2*ρ))*((sum(w) - 1.0)^2 + (transpose(ḡ)*w - R)^2);
 

"""
    function solve(model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem; 
        verbose::Bool = true, K::Int = 10000, T₀::Float64 = 1.0, T₁::Float64 = 0.1, 
        α::Float64 = 0.99, β::Float64 = 0.01, τ::Float64 = 0.99,
        μ::Float64 = 1.0, ρ::Float64 = 1.0) -> MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem

The `solve` function solves the minimum variance portfolio allocation problem using a simulated annealing approach for a given instance 
    of the [`MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`](@ref) problem type.

### Arguments
- `model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`: An instance of the [`MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`](@ref) that defines the problem parameters.
- `verbose::Bool = true`: A boolean flag to control verbosity of output during optimization.
- `K::Int = 10000`: The initial number of iterations at each temperature level.
- `T₀::Float64 = 1.0`: The initial temperature for the simulated annealing process.
- `T₁::Float64 = 0.1`: The final temperature for the simulated annealing process.
- `α::Float64 = 0.99`: The cooling rate for the temperature.
- `β::Float64 = 0.01`: The step size for generating new candidate solutions.
- `τ::Float64 = 0.99`: The penalty parameter update factor.
- `μ::Float64 = 1.0`: The initial penalty parameter for the logarithmic barrier term.
- `ρ::Float64 = 1.0`: The initial penalty parameter for the equality constraints.

### Returns
- `MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem`: The input model instance updated with the optimal portfolio weights.

"""
function solve(model::MySimulatedAnnealingMinimumVariancePortfolioAllocationProblem; 
    verbose::Bool = true, K::Int = 10000, T₀::Float64 = 1.0, T₁::Float64 = 0.1, 
    α::Float64 = 0.99, β::Float64 = 0.01, τ::Float64 = 0.99,
    μ::Float64 = 1.0, ρ::Float64 = 1.0)

    # initialize -
    has_converged = false;

    # unpack the model parameters -
    w = model.w;
    ḡ = model.ḡ;
    Σ̂ = model.Σ̂;
    R = model.R;

    # initialize parameters for simulated annealing -
    T = T₀; # initial T -
    current_w = w;
    current_f = _objective_function(current_w, ḡ, Σ̂, R, μ, ρ);
    
    # best solution found so far -
    w_best = current_w;
    f_best = current_f;
    KL = K;

    while has_converged == false
    
        accepted_counter = 0; 
        
        # TODO: Implement simulated annealing logic here -
         d = length(w)
+        min_eps = 1e-12
+
+        for k in 1:KL
+            # propose candidate by Gaussian perturbation
+            candidate = current_w .+ β .* randn(d)
+
+            # enforce a tiny positive floor (helps numerical stability / log barrier)
+            candidate = max.(candidate, min_eps)
+
+            # evaluate objective
+            f_candidate = _objective_function(candidate, ḡ, Σ̂, R, μ, ρ)
+
+            # acceptance: accept if better, otherwise with Metropolis probability
+        
+            if f_candidate <= current_f
+                accept = true
+            else
+                Δ = f_candidate - current_f
+                accept_prob = exp(-Δ / max(T, 1e-16))
+                accept = rand() < accept_prob
+            end
+
+            if accept
+                current_w = candidate
+                current_f = f_candidate
+                accepted_counter += 1
+
+                # record best
+                if current_f < f_best
+                    w_best = copy(current_w)
+                    f_best = current_f
+                end
+            end
+        end
+
+        # adapt KL based on acceptance rate
+        fraction_accepted = accepted_counter / max(KL, 1)
+        if fraction_accepted > 0.8
+            KL = max(10, ceil(Int, 0.75 * KL))
+        elseif fraction_accepted < 0.2
+            KL = ceil(Int, 1.5 * KL)
+        end
+
+        # update penalty parameters
+        μ *= τ
+        ρ *= τ
+
+        # cooling schedule
+        if T <= T₁
+            has_converged = true
+        else
+            T *= α
+        end
+
+        if verbose
+            @info "SA step" T=T KL=KL fraction_accepted=fraction_accepted f_best=f_best
+        end
# update model and return
    model.w = w_best
    return model
end


"""
    function solve(problem::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem) -> Dict{String,Any}

The `solve` function solves the Markowitz risky asset-only portfolio choice problem for a given instance of the [`MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem`](@ref) problem type.
The `solve` method checks for the optimization's status using an assertion. Thus, the optimization must be successful for the function to return.
Wrap the function call in a `try` block to handle exceptions.


### Arguments
- `problem::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem`: An instance of the [`MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem`](@ref) that defines the problem parameters.

### Returns
- `Dict{String, Any}`: A dictionary with optimization results.

The results dictionary has the following keys:
- `"reward"`: The reward associated with the optimal portfolio.
- `"argmax"`: The optimal portfolio weights.
- `"objective_value"`: The value of the objective function at the optimal solution.
- `"status"`: The status of the optimization.
"""
function solve(problem::MyMarkowitzRiskyAssetOnlyPortfolioChoiceProblem)::Dict{String,Any}

    # initialize -
    results = Dict{String,Any}()
    Σ = problem.Σ;
    μ = problem.μ;
    R = problem.R;
    bounds = problem.bounds;
    wₒ = problem.initial

    # setup the problem -
    d = length(μ)
    model = Model(()->MadNLP.Optimizer(print_level=MadNLP.ERROR, max_iter=500))
    @variable(model, bounds[i,1] <= w[i=1:d] <= bounds[i,2], start=wₒ[i])

    # set objective function -
    @objective(model, Min, transpose(w)*Σ*w);

    # setup the constraints -
    @constraints(model, 
        begin
            # my turn constraint
            transpose(μ)*w >= R
            sum(w) == 1.0
        end
    );

    # run the optimization -
    optimize!(model)

    # check: was the optimization successful?
    @assert is_solved_and_feasible(model)

    # populate -
    w_opt = value.(w);
    results["argmax"] = w_opt
    results["reward"] = transpose(μ)*w_opt; 
    results["objective_value"] = objective_value(model);
    results["status"] = termination_status(model);

    # return -
    return results
end