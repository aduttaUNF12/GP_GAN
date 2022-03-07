if isfile('SVMmodel.mat') % only trained model available
    load('SVMmodel.mat');
else % train the model and save it
    n = 32*32; % number of random numbers
    %false_trn_data = randi(21,3000,n)-11; % horizontal vector of 500 1x1025 fake samples with ranges between [-10,+10].
    false_trn_data = normrnd(0,1,[3000,n]);
    
    tamim_trn_data = load('C:\Users\n01388138\Documents\MATLAB\GAN tests\Training data\training data\Data + Images\SampleData32x32_3000.mat');
    true_trn_data = [];
    %rand500 = randi(1024,500,1);
    %true_trn_data_ = true_trn_data(1,1).TD;
    %true_trn_data_.TD = true_trn_data_{rand500,1};
    for c=1:3000
        TwoDArray = tamim_trn_data(1,1).TD{c,1}.test;
        OneDArray = reshape(TwoDArray.',1,[]);
        true_trn_data = vertcat(true_trn_data,OneDArray);
    end
    
    % final training data with random numbers from SSRR/ICRA papers (class=0) and
    % Training data used by our GAN model (class=1);
    X = vertcat(true_trn_data,false_trn_data);
    
    % generate labels for the training data
    labelsR = ones(3000,1);
    labelsG = zeros(3000,1);
    labels = vertcat(labelsR, labelsG);% this is training data label
    
    % train SVM
    Mdl = fitcsvm(X,labels);
    save SVMmodel.mat Mdl;
end

% test data is our GAN generated data
test_data = load('C:\Users\n01388138\Documents\MATLAB\GAN tests\generated data\GeneratedData_x1025_4Dsingles\Fixed_data_corrected_single_4Dform\New Data - Original Loss - NoF 30 - Sample Size 1025.mat');
%test_data = test_data(1,1).x(:,:,200,:);
epoch = 2;


% score plotting
% scorePlotter = animatedline();
% xlabel("Generated Data ID")
% ylabel("Score")
% grid on

correct = 0;
sampleSize = 1025;
for c = 1:sampleSize
    test2d = test_data(1,1).x(:,:,epoch,c);
    test1d = reshape(test2d.',1,[]);
    [label,score] = predict(Mdl,test1d);% correct label = 1 as it's supposed to be real-like and SVM should be fooled.
    if label == 1
        correct = correct+1;
        %addpoints(scorePlotter,c,score);
    end
end

accuracy = (correct/sampleSize)*100;
fprintf('SVM was fooled by the generated data %d percent of times!',accuracy);

%% plot training data
Zall = load('C:\Users\n01388138\Documents\MATLAB\GAN tests\Training data\training data\Data + Images\SampleData32x32_3000.mat');
ZMatrix = zeros(32,32,3000);
for c=1:3000
    Z = TD{c,1}.Z;
    Z = reshape(Z,32,32);
    Z = Z.';
    ZMatrix(:,:,c) = Z;
end

% plot here in a tile
figure();
filename = strcat('Training Data Samples');
t = tiledlayout(2,2,'TileSpacing','Compact');
title(t,filename);
for j = 1:1:4
    nexttile
    surf(ZMatrix(:,:,j),'EdgeColor','None');
    view(2);
    hold on
    set(gca,'xtick',[],'ytick',[])
    savefig("Generated Data of Epoch 200");
    
    scale=2;
    paperunits='centimeters';
    filewidth=7.5;%cm
    fileheight=5.5;%cm
    res=300;%resolution
    size=[filewidth fileheight]*scale;
    set(gcf,'paperunits',paperunits,'paperposition',[0 0 size]);
    set(gcf, 'PaperSize', size);
    %saveas(gcf,filename,'pdf');
end

% verify training data
Zt = tamim_trn_data(1,1).TD{1,1}.Z;
Zt = reshape(Zt,32,32);
Zt = Zt.';
Yt = tamim_trn_data(1,1).TD{1,1}.Y;
Yt = reshape(Yt,32,32);
Yt = Yt.';
used = tamim_trn_data(1,1).TD{1,1}.test;
figure(1);
surf(Zt); view(2);
title('Zt: No Noise');
figure(2);
surf(Yt); view(2);
title('Yt: With Noise');
figure(3);
surf(used); view(2);
title('Usedin Training');

%% plot generated data
load('32x32_3000 - FS 5 - NoF 32 - E 1000 - MBS 128 - Original Loss - Data.mat');
Z = gather(dlXGNArray);
for epoch = 1:1:10
    figure();
    filename = strcat('Generated Data (Epoch ',num2str(epoch*100),')');
    t = tiledlayout(2,2,'TileSpacing','Compact');
    title(t,filename);
    for j = 1:1:4
        nexttile
        surf(Z(:,:,epoch,j),'EdgeColor','None');
        view(2);
        hold on
        set(gca,'xtick',[],'ytick',[])
        savefig("Generated Data of Epoch 200");
        
        scale=2;
        paperunits='centimeters';
        filewidth=7.5;%cm
        fileheight=5.5;%cm
        res=300;%resolution
        size=[filewidth fileheight]*scale;
        set(gcf,'paperunits',paperunits,'paperposition',[0 0 size]);
        set(gcf, 'PaperSize', size);
        %saveas(gcf,filename,'pdf');
    end
end

