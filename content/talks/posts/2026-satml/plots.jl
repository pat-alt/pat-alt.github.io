using CounterfactualExplanations
using CounterfactualExplanations.Models
using Plots
using TaijaData
using TaijaPlotting

data = CounterfactualData(load_linearly_separable(25)...)
M = Model(MLP()) |> x -> x(data)
nepochs = 15

anim = @animate for i in 1:nepochs
        Models.forward!(M.model, Models.data_loader(data); loss=:logitcrossentropy, opt=flux_training_params.opt)

        plt = contourf(M, data;
                axis=nothing,
                cbar=false,
                legend=false,
                size=(400, 600),
                c=cgrad([RGBA(1, 0.80, 0, 0), RGBA(1, 0.30, 0, 1)]),
        )

        scatter!(plt, data.X[1, data.y[1, :].==0], data.X[2, data.y[1, :].==0];
                shape=:circle, color=:purple, ms=10, mscolor=:white,
        )
        scatter!(plt, data.X[1, data.y[1, :].==1], data.X[2, data.y[1, :].==1];
                shape=:diamond, color=:purple, ms=10, mscolor=:white,
        )
end

gif(anim, "www/images/training.gif"; fps=10)
savefig(plt, "www/images/training.png")

