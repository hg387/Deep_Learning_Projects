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



trainingdata(~isfinite(trainingdata)) = 0;
testingdata(~isfinite(testingdata)) = 0;


msg = sprintf('Wrong Input\n');

hiddenlayers = ((input('Enter the hidden layers variables:\n')));        
%while ((~isnumeric(hiddenlayers)) || (isempty(hiddenlayers)))
while ((~isnumeric(hiddenlayers)) || (isempty(hiddenlayers)) || ~(all(hiddenlayers(:) > 0)) || ~(all((fix(hiddenlayers) == hiddenlayers) == 1)))
    disp(msg);
    hiddenlayers = (input('Enter the hidden layers variables:\n')); 
end


numberOfThetas = size(hiddenlayers,2)+1;

%cell arrays for thetas and intermediate layer values
thetas = cell([numberOfThetas, 1]);
processValues = cell([(numberOfThetas+1),1]);
processValues{1,1} = trainingdata;
processValues{(numberOfThetas+1),1} = zeros(trainingSize,14);% only the last layer has zeros 

processValuesY = cell([(numberOfThetas+1),1]);
processValuesY{1,1} = testingdata;
processValuesY{(numberOfThetas+1),1} = zeros(testingSize,14); 

%initializing thetas to some random values b/w [0,1], then multiplying them
%with 0.01.
for i=1:numberOfThetas
    if i==1
        thetas{i,1} = (0.001).*rand(1601,hiddenlayers(1,i));
        processValues{i+1,1} = rand(trainingSize,hiddenlayers(1,i));
        processValuesY{i+1,1} = rand(testingSize,hiddenlayers(1,i));
    elseif i==numberOfThetas
        thetas{i,1} = (0.001).*rand(hiddenlayers(1,(i-1)),14);
    else
        thetas{i,1} = (0.001).*rand(hiddenlayers(1,(i-1)),hiddenlayers(1,i));
        processValues{i+1,1} = rand(trainingSize,hiddenlayers(1,i));
        processValuesY{i+1,1} = rand(testingSize,hiddenlayers(1,i));
    end
    
end
changeavglikelyhood = 0.1;
avgsY = [];
avgs = [];
iterations = [];
iteration = 0;
alpha = 0.00001;% L2 regularization amount
eta = 0.92;%learning rate
previousJ = 0;
previousJY = 0;

%Forward Initialization 
for i=1:(numberOfThetas+1)
   if (i==(numberOfThetas+1))
       gx = processValues{i,1};
       gxY = processValuesY{i,1};
       gx(~isfinite(gx)) = 0;
       gxY(~isfinite(gxY)) = 0;
       
       avgJ = (trainingY*(log(gx)')) + ((1.-trainingY)*(log(1.-gx)'));
       avgJY = (testingY*(log(gxY)')) + ((1.-testingY)*(log(1.-gxY)'));
       avgJ(~isfinite(avgJ)) = 0.0001;
       avgJY(~isfinite(avgJY)) = 0.0001;
       
       tj = trace(avgJ);
       tjY = trace(avgJY);
       changeavglikelyhood = (tj)/(double (trainingSize)); % initial J value
       previousJ = changeavglikelyhood;
       previousJY = (tjY)/(double (testingSize));
       break;
   end
   net = processValues{i,1} * thetas{i,1};
   netY = processValuesY{i,1} * thetas{i,1};
   net(~isfinite(net)) = 0;
   netY(~isfinite(netY)) = 0;
   
   nc = (1.+exp(((-1).*net)));
   gx = (1.0)./(1.+exp(((-1).*net)));
   gxY = (1.0)./(1.+exp(((-1).*netY)));
   gx(~isfinite(gx)) = 0;
   gxY(~isfinite(gxY)) = 0;
   
   processValues{i+1,1} = gx;
   processValuesY{i+1,1} = gxY;
end

tmpThetas = thetas;

%Forward-Backward Propagation
while (iteration < 20000)
    %Backward Propagation
    for i=(numberOfThetas):-1:1
        
        if i==((numberOfThetas))
            delta = (trainingY - processValues{end,1});
            deltaY = (testingY - processValuesY{end,1});
            dJ = (((processValues{i,1}')*(delta))) + (alpha.*tmpThetas{i,1});
            dJY = (((processValuesY{i,1}')*(deltaY))) + (alpha.*tmpThetas{i,1});
        else
            % Store previous thetas somewhere 
            tmp = ((delta*(tmpThetas{i+1,1}')) .* (processValues{i+1,1} .* (1.-processValues{i+1,1})));
            tmpY = ((deltaY*(tmpThetas{i+1,1}')) .* (processValuesY{i+1,1} .* (1.-processValuesY{i+1,1})));
            delta = tmp;
            deltaY = tmpY;
            dJ = ((processValues{i,1}')*delta) + (alpha.*tmpThetas{i,1});
            dJY = ((processValuesY{i,1}')*deltaY) + (alpha.*tmpThetas{i,1});
        end
        dJ(~isfinite(dJ)) = 0;
        dJY(~isfinite(dJY)) = 0;
        thetas{i,1} = thetas{i,1} + ((eta/double (trainingSize)).*dJ);
    end
    
    tmpThetas = thetas;
    
    %Forward Propagation
    for j=1:(numberOfThetas)
        net = processValues{j,1} * thetas{j,1};
        netY = processValuesY{j,1} * thetas{j,1};
        net(~isfinite(net)) = 0;
        netY(~isfinite(netY)) = 0;
        
        gx = (1.0)./(1.+exp(((-1).*net)));
        gxY = (1.0)./(1.+exp(((-1).*netY)));
        gx(~isfinite(gx)) = 0;
        gxY(~isfinite(gxY)) = 0;
        
        processValues{j+1,1} = gx;
        processValuesY{j+1,1} = gxY;
       
    end
    
    iteration = iteration + 1;
    iterations(iteration,1) = iteration;
    
    recA = (log(processValues{end,1})');
    recA(~isfinite(recA)) = 0.0001;
    recB = (log(1.-processValues{end,1})');
    recB(~isfinite(recB)) = 0.0001;
    avgJ = (trainingY*(recA)) + ((1.-trainingY)*(recB));
    
    recAY = (log(processValuesY{end,1})');
    recAY(~isfinite(recAY)) = 0.0001;
    recBY = (log(1.-processValuesY{end,1})');
    recBY(~isfinite(recBY)) = 0.0001;
    avgJY = (testingY*(recAY)) + ((1.-testingY)*(recBY));
    avgJ(~isfinite(avgJ)) = 0.0001;
    avgJY(~isfinite(avgJY)) = 0.0001;
        
    % add in L2 for all the layers
    tj = 0;
    tjY = 0;
    for m=1:numberOfThetas
        L2 = ((alpha).*(log((thetas{m,1}')*(thetas{m,1}))));
        L2(~isfinite(L2)) = 0.0001;
        tj = tj + trace(L2);
        tjY = tjY + trace(L2);
    end
    tj = tj + trace(avgJ); 
    tjY = tjY + trace(avgJY);
    sumavgJ = (tj)/(double (trainingSize));
    sumavgJY = (tjY)/(double (testingSize));
    changeavglikelyhood = sumavgJ - previousJ;
    previousJ = sumavgJ;
    
    avgs(iteration,1) = sumavgJ;
    avgsY(iteration,1) = sumavgJY;
end

% Accuracy Calculation
TP = 0;
newY = zeros(size(testingY,1),size(testingY,2));

[maxE, I] = max(gxY,[],2);
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
savefig('ANN3.fig');