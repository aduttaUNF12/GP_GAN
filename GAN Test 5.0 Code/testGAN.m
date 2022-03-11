%% Load Training Data
% url = "http://download.tensorflow.org/example_images/flower_photos.tgz";
% downloadFolder = tempdir;
% filename = fullfile(downloadFolder,"flower_dataset.tgz");
%
% imageFolder = fullfile(downloadFolder,"flower_photos");
% if ~exist(imageFolder,"dir")
%     disp("Downloading Flowers data set (218 MB)...")
%     websave(filename,url);
%     untar(filename,downloadFolder)
% end

% datasetFolder = fullfile(imageFolder);

%32x32_3000 - FS 1,3 - NoF 10, 20
%convert dlarray to matrix
%add grayscale image to original figure (include color bar by default)

clc
clear

for first = 3
    for second = 2
        for third = 3
            for fourth = 4

                if first == 1
                    loadFile1 = "8x8_";
                    IS = 8;
                elseif first == 2
                    loadFile1 = "16x16_";
                    IS = 16;
                else
                    loadFile1 = "32x32_";
                    IS = 32;
                end

                if second == 1
                    loadFile2 = 1000;
                elseif second == 2
                    loadFile2 = 3000;
                else
                    loadFile2 = 5000;
                end

                if third == 1
                    FS = 1;
                elseif third == 2
                    FS = 3;
                else
                    FS = 5;
                end

                if fourth == 1
                    NoF = 10;
                elseif fourth == 2
                    NoF = 20;
                elseif fourth == 3
                    NoF = 30;
                else
                    if first == 1
                        NoF = 8;
                    elseif first == 2
                        NoF = 16;
                    else
                        NoF = 32;
                    end
                end

                % workaround = load("SampleData" + loadFile1 + loadFile2 + ".mat", "TD");
                workaround = load("gansM32-N32-B1-L1-D3000-StructuredMean.mat", "gans");

                % imds = workaround.TD{1, 1}.test;

                for xyz = 1:loadFile2
                    if xyz == 1
                        placeholder = workaround.gans.TD{1, 1}.Z;
                        imds = rot90(reshape(placeholder, [], 32));
                        % imds = rescale(imds,0,1); % for rescaling the input to [0,1]
                    else
                        placeholder = workaround.gans.TD{xyz, 1}.Z;
                        placeholder = rot90(reshape(placeholder, [], 32));
                        % placeholder = rescale(placeholder,0,1);
                        imds = cat(4, imds, placeholder);
                    end
                end

                augmenter = imageDataAugmenter(RandXReflection=false);
                augimds = augmentedImageDatastore([IS IS],imds,DataAugmentation=augmenter);

                %% Define Generator Network
                %matlab function that visualizes input layer
                filterSize = FS;
                numFilters = NoF;
                numLatentInputs = 100;

                projectionSize = [4 4 512];

                if IS == 32
                    layersGenerator = [
                        featureInputLayer(numLatentInputs,Normalization="none")
                        fullyConnectedLayer(prod(projectionSize))
                        functionLayer(@(X) feature2image(X,projectionSize),Formattable=true)
                        %         transposedConv2dLayer(filterSize,4*numFilters)
                        %         batchNormalizationLayer
                        %         reluLayer
                        transposedConv2dLayer(filterSize,2*numFilters,Stride=2,Cropping="same")
                        batchNormalizationLayer
                        reluLayer
                        transposedConv2dLayer(filterSize,numFilters,Stride=2,Cropping="same")
                        batchNormalizationLayer
                        reluLayer
                        transposedConv2dLayer(filterSize,1,Stride=2,Cropping="same")
                        tanhLayer];
                elseif IS == 16
                    layersGenerator = [
                        featureInputLayer(numLatentInputs)
                        fullyConnectedLayer(prod(projectionSize))
                        functionLayer(@(X) feature2image(X,projectionSize),Formattable=true)
                        %         transposedConv2dLayer(filterSize,4*numFilters)
                        %         batchNormalizationLayer
                        %         reluLayer
                        %         transposedConv2dLayer(filterSize,2*numFilters,Stride=2,Cropping="same")
                        %         batchNormalizationLayer
                        %         reluLayer
                        transposedConv2dLayer(filterSize,numFilters,Stride=2,Cropping="same")
                        batchNormalizationLayer
                        reluLayer
                        transposedConv2dLayer(filterSize,1,Stride=2,Cropping="same")
                        tanhLayer];
                else
                    layersGenerator = [
                        featureInputLayer(numLatentInputs)
                        fullyConnectedLayer(prod(projectionSize))
                        functionLayer(@(X) feature2image(X,projectionSize),Formattable=true)
                        %         transposedConv2dLayer(filterSize,4*numFilters)
                        %         batchNormalizationLayer
                        %         reluLayer
                        %         transposedConv2dLayer(filterSize,2*numFilters,Stride=2,Cropping="same")
                        %         batchNormalizationLayer
                        %         reluLayer
                        %         transposedConv2dLayer(filterSize,numFilters,Stride=2,Cropping="same")
                        %         batchNormalizationLayer
                        %         reluLayer
                        transposedConv2dLayer(filterSize,1,Stride=2,Cropping="same")
                        tanhLayer];
                end

                lgraphGenerator = layerGraph(layersGenerator);
                dlnetGenerator = dlnetwork(lgraphGenerator);


                %% Define Discriminator Network
                dropoutProb = 0.5;
                numFilters = NoF;
                scale = 0.2;

                inputSize = [IS IS 1];
                filterSize = FS;

                if IS == 32
                    layersDiscriminator = [
                        imageInputLayer(inputSize,Normalization="none")
                        dropoutLayer(dropoutProb)
                        %         convolution2dLayer(filterSize,numFilters,Stride=2,Padding="same")
                        %         leakyReluLayer(scale)
                        convolution2dLayer(filterSize,2*numFilters,Stride=2,Padding="same")
                        batchNormalizationLayer
                        leakyReluLayer(scale)
                        convolution2dLayer(filterSize,4*numFilters,Stride=2,Padding="same")
                        batchNormalizationLayer
                        leakyReluLayer(scale)
                        convolution2dLayer(filterSize,8*numFilters,Stride=2,Padding="same")
                        batchNormalizationLayer
                        leakyReluLayer(scale)
                        convolution2dLayer(4,1)];
                elseif IS == 16
                    layersDiscriminator = [
                        imageInputLayer(inputSize,Normalization="none")
                        dropoutLayer(dropoutProb)
                        %         convolution2dLayer(filterSize,numFilters,Stride=2,Padding="same")
                        %         leakyReluLayer(scale)
                        %         convolution2dLayer(filterSize,2*numFilters,Stride=2,Padding="same")
                        %         batchNormalizationLayer
                        %         leakyReluLayer(scale)
                        convolution2dLayer(filterSize,4*numFilters,Stride=2,Padding="same")
                        batchNormalizationLayer
                        leakyReluLayer(scale)
                        convolution2dLayer(filterSize,8*numFilters,Stride=2,Padding="same")
                        batchNormalizationLayer
                        leakyReluLayer(scale)
                        convolution2dLayer(4,1)];
                else
                    layersDiscriminator = [
                        imageInputLayer(inputSize,Normalization="none")
                        dropoutLayer(dropoutProb)
                        %         convolution2dLayer(filterSize,numFilters,Stride=2,Padding="same")
                        %         leakyReluLayer(scale)
                        %         convolution2dLayer(filterSize,2*numFilters,Stride=2,Padding="same")
                        %         batchNormalizationLayer
                        %         leakyReluLayer(scale)
                        %         convolution2dLayer(filterSize,4*numFilters,Stride=2,Padding="same")
                        %         batchNormalizationLayer
                        %         leakyReluLayer(scale)
                        convolution2dLayer(filterSize,8*numFilters,Stride=2,Padding="same")
                        batchNormalizationLayer
                        leakyReluLayer(scale)
                        convolution2dLayer(4,1)];
                end

                lgraphDiscriminator = layerGraph(layersDiscriminator);
                dlnetDiscriminator = dlnetwork(lgraphDiscriminator);

                % analyzeNetwork(layersGenerator);
                % analyzeNetwork(layersDiscriminator);

                %% Specify Training Options
                numEpochs = 1000;
                miniBatchSize = 128;

                learnRate = 0.0002;
                gradientDecayFactor = 0.5;
                squaredGradientDecayFactor = 0.999;

                flipFactor = 0.3;

                validationFrequency = 200;


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

                numValidationImages = 1025;
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
                figNum = 0;

                % Loop over epochs.


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
                            image(imageAxes,I)
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
                    dlXGeneratedNew = predict(dlnetGenerator,dlZValidation);

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

            end
        end
    end
end

fprintf("Success!\n");