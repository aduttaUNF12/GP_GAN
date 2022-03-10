function [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
    modelGradients(dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor)

% Calculate the predictions for real data with the discriminator network.
dlYPred = forward(dlnetDiscriminator, dlX);

% Calculate the predictions for generated data with the discriminator network.
[dlXGenerated,stateGenerator] = forward(dlnetGenerator,dlZ);
dlYPredGenerated = forward(dlnetDiscriminator, dlXGenerated);

% Convert the discriminator outputs to probabilities.
probGenerated = sigmoid(dlYPredGenerated);
probReal = sigmoid(dlYPred);

% Calculate the score of the discriminator.
scoreDiscriminator = (mean(probReal) + mean(1-probGenerated)) / 2;

try
    load("ScoreDisc - Original Loss - NoF 30", 'scoreDiscArr');
catch
    scoreDiscArr = -1;
end
if size(scoreDiscArr) == [23000 1]
    try
        load("ScoreDisc - Original Loss - NoF 32", 'scoreDiscArr');
        scoreDiscArr = cat(1, scoreDiscArr, extractdata(scoreDiscriminator));
    catch
        scoreDiscArr = extractdata(scoreDiscriminator);
    end
    save("ScoreDisc - Original Loss - NoF 32", "scoreDiscArr");

else
    try
        load("ScoreDisc - Original Loss - NoF 30", 'scoreDiscArr');
        scoreDiscArr = cat(1, scoreDiscArr, extractdata(scoreDiscriminator));
    catch
        scoreDiscArr = extractdata(scoreDiscriminator);
    end
    save("ScoreDisc - Original Loss - NoF 30", "scoreDiscArr");
end

% Calculate the score of the generator.
scoreGenerator = mean(probGenerated);

try
    load("ScoreGen - Original Loss - NoF 30", 'scoreGenArr');
catch
    scoreGenArr = -1;
end
if size(scoreGenArr) == [23000 1]
    try
        load("ScoreGen - Original Loss - NoF 32", 'scoreGenArr');
        scoreGenArr = cat(1, scoreGenArr, extractdata(scoreGenerator));
    catch
        scoreGenArr = extractdata(scoreGenerator);
    end
    save("ScoreGen - Original Loss - NoF 32", "scoreGenArr");
else
    try
        load("ScoreGen - Original Loss - NoF 30", 'scoreGenArr');
        scoreGenArr = cat(1, scoreGenArr, extractdata(scoreGenerator));
    catch
        scoreGenArr = extractdata(scoreGenerator);
    end
    save("ScoreGen - Original Loss - NoF 30", "scoreGenArr");
end

% Randomly flip a fraction of the labels of the real images.
numObservations = size(probReal,4);
idx = randperm(numObservations,floor(flipFactor * numObservations));

% Flip the labels.
probReal(:,:,:,idx) = 1 - probReal(:,:,:,idx);

% Calculate the GAN loss.
[lossGenerator, lossDiscriminator] = ganLoss(probReal,probGenerated);

try
    load("LossGen - Original Loss - NoF 30", 'lossGenArr');
catch
    lossGenArr = -1;
end
if size(lossGenArr) == [23000 1]
    try
        load("LossGen - Original Loss - NoF 32", 'lossGenArr');
        lossGenArr = cat(1, lossGenArr, extractdata(lossGenerator));
    catch
        lossGenArr = extractdata(lossGenerator);
    end
    save("LossGen - Original Loss - NoF 32", "lossGenArr");
else
    try
        load("LossGen - Original Loss - NoF 30", 'lossGenArr');
        lossGenArr = cat(1, lossGenArr, extractdata(lossGenerator));
    catch
        lossGenArr = extractdata(lossGenerator);
    end
    save("LossGen - Original Loss - NoF 30", "lossGenArr");
end

try
    load("LossDisc - Original Loss - NoF 30", 'lossDiscArr');
catch
    lossDiscArr = -1;
end
if size(lossDiscArr) == [23000 1]
    try
        load("LossDisc - Original Loss - NoF 32", 'lossDiscArr');
        lossDiscArr = cat(1, lossDiscArr, extractdata(lossDiscriminator));
    catch
        lossDiscArr = extractdata(lossDiscriminator);
    end
    save("LossDisc - Original Loss - NoF 32", "lossDiscArr");
else
    try
        load("LossDisc - Original Loss - NoF 30", 'lossDiscArr');
        lossDiscArr = cat(1, lossDiscArr, extractdata(lossDiscriminator));
    catch
        lossDiscArr = extractdata(lossDiscriminator);
    end
    save("LossDisc - Original Loss - NoF 30", "lossDiscArr");
end

% For each network, calculate the gradients with respect to the loss.
gradientsGenerator = dlgradient(lossGenerator, dlnetGenerator.Learnables,RetainData=true);
gradientsDiscriminator = dlgradient(lossDiscriminator, dlnetDiscriminator.Learnables);

end