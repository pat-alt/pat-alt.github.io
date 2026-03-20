using CounterfactualExplanations
using CounterfactualExplanations.Models
using DifferentiationInterface
using Optimisers
using Plots
import Random
using TaijaData
using TaijaPlotting

data = CounterfactualData(load_linearly_separable(25)...)
M = Model(MLP()) |> x -> x(data)
nepochs = 15

plts = []
anim = @animate for i in 1:nepochs
        Models.forward!(M.model, Models.data_loader(data); loss=:logitcrossentropy, opt=flux_training_params.opt)

        plt = contourf(M, data;
                axis=nothing,
                cbar=false,
                legend=false,
                size=(400, 600),
                c=:Wistia,
        )

        scatter!(plt, data.X[1, data.y[1, :].==0], data.X[2, data.y[1, :].==0];
                shape=:circle, color=:purple, ms=8, mscolor=:purple,
        )
        scatter!(plt, data.X[1, data.y[1, :].==1], data.X[2, data.y[1, :].==1];
                shape=:diamond, color=:purple, ms=8, mscolor=:purple,
        )

        if i == nepochs
                push!(plts, plt)
        end
end

gif(anim, "www/images/training.gif"; fps=8)
savefig(plts[1], "www/images/training.png")

Random.seed!(123)
generator = GenericGenerator(opt=Descent(0.005))
x = select_factual(data, rand(findall(data.output_encoder.labels .== 1)))
set_global_ad_backend(AutoForwardDiff())
ce = generate_counterfactual(x, 2, data, M, generator)
xs = reduce(hcat, path(ce))

nsteps = size(xs, 2)
plts = []

anim_ce = @animate for i in 1:nsteps
        plt = contourf(M, data;
                axis=nothing,
                cbar=false,
                legend=false,
                size=(400, 600),
                c=:Wistia,
        )

        scatter!(plt, data.X[1, data.y[1, :].==0], data.X[2, data.y[1, :].==0];
                shape=:circle, color=:purple, ms=8, mscolor=:purple,
        )
        scatter!(plt, data.X[1, data.y[1, :].==1], data.X[2, data.y[1, :].==1];
                shape=:diamond, color=:purple, ms=8, mscolor=:purple,
        )

        # Plot the path up to step i
        plot!(plt, xs[1, 1:i], xs[2, 1:i];
                color=:purple, lw=5, label=nothing,
        )

        # Highlight current position
        scatter!(plt, [xs[1, i]], [xs[2, i]];
                shape=:star5, color=:purple, ms=15, mscolor=:purple, label=nothing,
        )

        # Mark the starting factual
        scatter!(plt, [xs[1, 1]], [xs[2, 1]];
                shape=:diamond, color=:purple, ms=15, mscolor=:purple, label=nothing,
        )

        if i == nsteps
                push!(plts, plt)
        end
end

gif(anim_ce, "www/images/counterfactual.gif"; fps=25)
savefig(plts[1], "www/images/counterfactual.png")
