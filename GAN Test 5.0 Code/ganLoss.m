function [lossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated)

n = 4;
x = randsample(n,1);

    if x == 1 % Matlab's original loss function, negative log liklihood
    % Calculate the loss for the generator network.
    lossGenerator = -mean(log(probGenerated));

    elseif x == 2% minmax from the paper
    lossGenerator = 0.5 * mean(log(1-probReal));

    elseif x == 3 % Least-Square form the paper
    lossGenerator = mean (log((probReal-1).^2));

    % Heuristic loss from the paper
    elseif x == 4 % Least-Square form the paper
    lossGenerator = 0.5 * mean(log(probReal));
    end

    % Calculate the loss for the discriminator network. negative log liklihood
    lossDiscriminator = -mean(log(probReal)) - mean(log(1-probGenerated)); 

end