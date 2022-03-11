%% augimds does reflection, why?
%% redundant code.
%% image input layer in the discriminator, why?
%% validation frequency is 256, why?

imds = zeros(1024,3000);
for c=1:3000
    Z = gans.TD{c,1}.Z;
    imds(:,c) = Z; 
end

augmenter = imageDataAugmenter(RandXReflection=false);
augimds = augmentedImageDatastore([1024 1],imds,DataAugmentation=augmenter);


numLatentInputs = 100;

projectionSize = [1 2048];


layersGenerator = [
    featureInputLayer(numLatentInputs)
    fullyConnectedLayer(prod(projectionSize))
    reluLayer
    fullyConnectedLayer(prod(projectionSize))
    reluLayer
    fullyConnectedLayer(prod(projectionSize)/2)
    reluLayer
    ];

lgraphGenerator = layerGraph(layersGenerator);
dlnetGenerator = dlnetwork(lgraphGenerator);

analyzeNetwork(dlnetGenerator);

inputSize = [1024 1];
layersDiscriminator = [
    featureInputLayer(prod(inputSize),Normalization="none")
    fullyConnectedLayer(prod(inputSize))
    reluLayer
    fullyConnectedLayer(prod(inputSize))
    reluLayer
    %fullyConnectedLayer(prod(inputSize))
    %reluLayer
    fullyConnectedLayer(1)
    sigmoidLayer
    ];

lgraphDiscriminator = layerGraph(layersDiscriminator);
dlnetDiscriminator = dlnetwork(lgraphDiscriminator);

analyzeNetwork(dlnetDiscriminator);

%% Specify Training Options
numEpochs = 500;
miniBatchSize = 128;

learnRate = 0.0002;
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;

flipFactor = 0.3;

validationFrequency = 100;


%% Train Model
augimds.MiniBatchSize = miniBatchSize;

executionEnvironment = "auto";

mbq = minibatchqueue(augimds,...
    MiniBatchSize=miniBatchSize,...
    PartialMiniBatch="discard",...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat="SSCB",...
    OutputEnvironment=executionEnvironment);

trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
trailingAvgDiscriminator = [];
trailingAvgSqDiscriminator = [];

numValidationImages = 25;
ZValidation = randn(numLatentInputs,numValidationImages,"single");

dlZValidation = dlarray(ZValidation,"CB");

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZValidation = gpuArray(dlZValidation);
end

f = figure;
f.Position(3) = 2*f.Position(3);

imageAxes = subplot(1,2,1);
scoreAxes = subplot(1,2,2);

lineScoreGenerator = animatedline(scoreAxes,Color=[0 0.447 0.741]);
lineScoreDiscriminator = animatedline(scoreAxes,Color=[0.85 0.325 0.098]);
legend("Generator","Discriminator");
ylim([0 1])
xlabel("Iteration")
ylabel("Score")
grid on

iteration = 0;
start = tic;



for epoch = 1:numEpochs

    % Reset and shuffle datastore.
    shuffle(mbq);

    % Loop over mini-batches.
    while hasdata(mbq)
        iteration = iteration + 1;

        % Read mini-batch of data.
        dlX = next(mbq);

        % Generate latent inputs for the generator network. Convert to
        % dlarray and specify the dimension labels "CB" (channel,n batch).
        % If training on a GPU, then convert latent inputs to gpuArray.
        Z = randn(numLatentInputs,miniBatchSize,"single");
        dlZ = dlarray(Z,"CB");

        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlZ = gpuArray(dlZ);
        end

        % Evaluate the model gradients and the generator state using
        % dlfeval and the modelGradients function listed at the end of the
        % example.
        [gradientsGenerator, gradientsDiscriminator, stateGenerator, scoreGenerator, scoreDiscriminator] = ...
            dlfeval(@modelGradients, dlnetGenerator, dlnetDiscriminator, dlX, dlZ, flipFactor);
        dlnetGenerator.State = stateGenerator;

        % Update the discriminator network parameters.
        [dlnetDiscriminator,trailingAvgDiscriminator,trailingAvgSqDiscriminator] = ...
            adamupdate(dlnetDiscriminator, gradientsDiscriminator, ...
            trailingAvgDiscriminator, trailingAvgSqDiscriminator, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Update the generator network parameters.
        [dlnetGenerator,trailingAvgGenerator,trailingAvgSqGenerator] = ...
            adamupdate(dlnetGenerator, gradientsGenerator, ...
            trailingAvgGenerator, trailingAvgSqGenerator, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Every validationFrequency iterations, display batch of generated images using the
        % held-out generator input.
        if mod(iteration,validationFrequency) == 0 || iteration == 1
            % Generate images using the held-out generator input.
            dlXGeneratedValidation = predict(dlnetGenerator,dlZValidation);

            % Tile and rescale the images in the range [0 1].
            I = imtile(extractdata(dlXGeneratedValidation));
            I = rescale(I);

            % Display the images.
            subplot(1,2,1);
            imshow(imageAxes,I)
            xticklabels([]);
            yticklabels([]);
            title("Generated Images");
        end

        % Update the scores plot.
        subplot(1,2,2)
        addpoints(lineScoreGenerator,iteration,...
            double(gather(extractdata(scoreGenerator))));

        addpoints(lineScoreDiscriminator,iteration,...
            double(gather(extractdata(scoreDiscriminator))));

        % Update the title with training progress information.
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        title(...
            "Epoch: " + epoch + ", " + ...
            "Iteration: " + iteration + ", " + ...
            "Elapsed: " + string(D))

        drawnow
    end

    figNum = figNum + 1;
    dlXGeneratedNew = predict(dlnetGenerator,dlZNew);

    if epoch == 100
        timeSave = duration(0,0,toc(start),Format="hh:mm:ss");
        dlXGNArray = extractdata(dlXGeneratedNew);
        genModel = dlnetGenerator;
        discModel = dlnetDiscriminator;
    elseif mod(epoch, 100) == 0
        timeSave = cat(1, timeSave, duration(0,0,toc(start),Format="hh:mm:ss"));
        dlXGNArray = cat(3, dlXGNArray, extractdata(dlXGeneratedNew));
        genModel = cat(1, genModel, dlnetGenerator);
        discModel = cat(1, discModel, dlnetDiscriminator);
    end

    %     if(mod(figNum, 10) == 0)
    %         numObservations = 25;
    %         ZNew = randn(numLatentInputs,numObservations,"single");
    %         dlZNew = dlarray(ZNew,"CB");
    %
    %         if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    %             dlZNew = gpuArray(dlZNew);
    %         end
    %
    %         dlXGeneratedNew = predict(dlnetGenerator,dlZNew);
    %
    %         I = imtile(extractdata(dlXGeneratedNew));
    %         I = rescale(I);
    %         y = figure;
    %         image(I)
    %         axis off
    %         title("Generated Images")
    %         savefig(numEpochs + " Total Epochs - Epoch " + figNum + " out of " + numEpochs);
    %         close(y);
    %     end
end


%% Generate New Images
numObservations = 1025;
ZNew = randn(numLatentInputs,numObservations,"single");
dlZNew = dlarray(ZNew,"CB");

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    dlZNew = gpuArray(dlZNew);
end

dlXGeneratedNew = predict(dlnetGenerator,dlZNew);
I = imtile(extractdata(dlXGeneratedNew));
I = rescale(I);
figure;
image(I)
axis off
title("Generated Images")
% savefig(numEpochs + " Total Epochs - Final Images")
% savefig(f, numEpochs + " Total Epochs - Final Results")

savefig(f, loadFile1 + loadFile2 + " - FS " + FS + " - NoF " + NoF + " - E 1000 - MBS 128 - Original Loss - Graph")

%dlXGNArray = cat(3, dlXGNArray, extractdata(dlXGeneratedNew));
save("Time - Original Loss - NoF " + NoF, "timeSave");
save(loadFile1 + loadFile2 + " - FS " + FS + " - NoF " + NoF + " - E 1000 - MBS 128 - Original Loss - Data", "dlXGNArray");
save("GenModel - Original Loss - NoF " + NoF, "genModel");
save("DiscModel - Original Loss - NoF " + NoF, "discModel");



fprintf("Success!\n");