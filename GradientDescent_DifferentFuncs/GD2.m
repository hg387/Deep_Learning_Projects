clear all; %remove all the old variables in the workspace
close all;

%if database already exists
if exist('database.mat', 'file') == 2
    load('database.mat')
else
    files = dir(fullfile('yalefaces','*subject*'));%replace yalefaces with name of the dir here
    if isempty(files)
        error('dir yalefaces not exits or empty');
    end
    
    data = zeros(154,1600);%preallocating data matrix
    testingdata = zeros(154,1);

    for i=1:length(files)
        tmp=imread(fullfile('yalefaces',files(i).name));%read files
        subjectclass = sscanf(files(i).name,'subject%d*');
        
        testingdata(i,1) = subjectclass;
        file=imresize(tmp, [40, 40]);%compress them to 40 by 40
        final = reshape(file, [1, 1600]);%flatten them to 1 by 1600
        data(i,:) = final;%concatenate to data matrix
    end

    save('database.mat','data','testingdata');%saving the data matrix in file
    clearvars -except data testingdata
end

data = [data testingdata];
rng(2,'philox');
randomdata = data(randperm(size(data(:,:,1), 1)),:);

data = randomdata(:,1:end-1);
Y = zeros(size(data, 1),14);
for i=1:size(data, 1)
    Y(i, (randomdata(i,end)-1)) = 1;
end

trainingSize = uint32((2/3)*size(data,1));
testingSize = size(data,1) - trainingSize;

trainingdata = data(1:trainingSize,:);
testingdata = data(trainingSize+1:end,:);

trainingY = Y(1:trainingSize,:);
testingY = Y(trainingSize+1:end,:);

m = mean(trainingdata(:,:));
s = std(trainingdata(:,:));
trainingdata(:,:)  = trainingdata(:,:) - repmat(m,size(trainingdata,1),1);
trainingdata(:,:) = trainingdata(:,:)./repmat(s,size(trainingdata,1),1);

testingdata(:,:)  = testingdata(:,:) - repmat(m,size(testingdata,1),1);
testingdata(:,:) = testingdata(:,:)./repmat(s,size(testingdata,1),1);

bias = ones(size(trainingdata,1),1);
trainingdata = [bias, trainingdata];

bias = ones(size(testingdata,1),1);
testingdata = [bias, testingdata];

%thetas = randi([-1 1],1601,14);
thetas = rand(1601,14);

trainingdata(~isfinite(trainingdata)) = 0;
testingdata(~isfinite(testingdata)) = 0;

eta = 0.45;
changeavglikelyhood = 0.1;
avgsY = [];
avgs = [];
iterations = [];
iteration = 0;
alpha = 0.001;
previousJ = 0;

while (abs(changeavglikelyhood) > 2^(-24)) && (iteration < 200)
    iteration = iteration + 1;
    iterations(iteration,1) = iteration;
    
    N = trainingSize;
    NY = testingSize;
    gx = (1.0)./(1.+exp(((-1).*(trainingdata(:,:)*thetas(:,:)))));
    gx(~isfinite(gx)) = 0;
    
    dJ = (((trainingdata(:,:)')*(trainingY - gx))) + (alpha.*thetas(:,:));
    thetas(:,:) = thetas(:,:)+((eta/double (N)).*dJ);
    gx = (1.0)./(1.+exp(((-1).*(trainingdata(:,:)*thetas(:,:)))));
    gx(~isfinite(gx)) = 0;
    
    avgJ = (trainingY*(log(gx)')) + ((1.-trainingY)*(log(1.-gx)'));
    avgJ(~isfinite(avgJ)) = 0;
    
    L2 = (2.*(log((thetas')*(thetas))));
    tj = trace(avgJ) + trace(L2);
    sumavgJ = (tj)/(double (N));
    changeavglikelyhood = sumavgJ - previousJ;
    previousJ = sumavgJ;
    
    avgs(iteration,1) = sumavgJ;
    
    % Avg log likelihood calculation of testing classes
    gxY = (1.0)./(1.+exp(((-1).*(testingdata(:,:)*thetas(:,:)))));
    gxY(~isfinite(gxY)) = 0;
    
    avgJY = (testingY*(log(gxY)')) + ((1.-testingY)*(log(1.-gxY)'));
    avgJY(~isfinite(avgJY)) = 0;
    
    tjY = trace(avgJY) + trace(L2);
    sumavgJY = (tjY)/(double (NY));
    
    avgsY(iteration,1) = sumavgJY;
end


TP = 0;
newY = zeros(size(testingY,1),size(testingY,2));
gx = (1.0)./(1.+exp(((-1).*(testingdata(:,:)*thetas(:,:)))));
gx(~isfinite(gx)) = 0;

[maxE, I] = max(gx,[],2);
for i=1:size(newY,1)
    newY(i,I(i,1)) = 1;
    if (testingY(i,I(i,1)) == 1)
        TP = TP + 1;
    end       
end
   
Accuracy =  (double (TP)/double (testingSize));
disp('Accuracy: ');
disp(Accuracy);

subplot(2,2,1);
plot(iterations, avgs);
title('For Training Set')
xlabel('Iteration Number');
ylabel('Average Log Likelihood');

subplot(2,2,2);
plot(iterations, avgsY);
title('For Testing Set')
xlabel('Iteration Number');
ylabel('Average Log Likelihood');

confusionMatrix = (testingY')*(newY);
%disp(confusionMatrix);
subplot(2,2,[3,4]);
imagesc(confusionMatrix);
title('Confusion Matrix');
for a=1:(size(confusionMatrix,1))
    for b=1:(size(confusionMatrix,2))
        text(b,a,num2str(confusionMatrix(a,b)))
    end
end
savefig('GD2.fig');