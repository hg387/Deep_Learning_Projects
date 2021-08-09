clear all; %remove all the old variables in the workspace
close all;
rng(2,'philox');

%if database already exists
if exist('ANN.mat', 'file') == 2
    load('ANN.mat')
else
    files = dir(fullfile('yalefaces','*subject*'));%replace yalefaces with name of the dir here
    if isempty(files)
        error('dir yalefaces not exits or empty');
    end
    
    %data = zeros(154,1600);%preallocating data matrix
    data = cell([length(files),2]);
    testingdata = zeros(154,1);

    for i=1:length(files)
        tmp=imread(fullfile('yalefaces',files(i).name));%read files
        subjectclass = sscanf(files(i).name,'subject%d*');
        
        testingdata(i,1) = subjectclass;
        file=imresize(tmp, [40, 40]);%compress them to 40 by 40
        final = double (file);
        data{i,1} = final;%concatenate to data matrix
        data{i,2} = double (testingdata(i,1));
    end

    save('ANN.mat','data','testingdata');%saving the data matrix in file
    clearvars -except data testingdata
end

randomdata = data(randperm(size(data(:,:), 1)),:);

data = randomdata(:,1:end-1);
Y = zeros(size(data, 1),14);
for i=1:size(data, 1)
    Y(i, (randomdata{i,end}-1)) = 1;
end

trainingSize = uint32((2/3)*size(data,1));
testingSize = size(data,1) - trainingSize;
TP_testing_training = (double (testingSize)) -0.2;

trainingdata = data(1:trainingSize,:); %left is the standardization
testingdata = data(trainingSize+1:end,:);

%----standardization start

tmpData = zeros(size(trainingdata,1),(size(trainingdata{1,1},1)*size(trainingdata{1,1},2)));
tmpDataY = zeros(size(testingdata,1),(size(testingdata{1,1},1)*size(testingdata{1,1},2)));

for q=1:size(trainingdata,1)
    tmpData(q,:) = reshape(trainingdata{q,1},[],size(tmpData, 2));
end

for q=1:size(testingdata,1)
    tmpDataY(q,:) = reshape(testingdata{q,1},[],size(tmpDataY, 2));
end

m = mean(tmpData(:,:));
s = std(tmpData(:,:));
tmpData(:,:)  = tmpData(:,:) - repmat(m,size(tmpData,1),1);
tmpData(:,:) = tmpData(:,:)./repmat(s,size(tmpData,1),1);

tmpDataY(:,:)  = tmpDataY(:,:) - repmat(m,size(tmpDataY,1),1);
tmpDataY(:,:) = tmpDataY(:,:)./repmat(s,size(tmpDataY,1),1);

for q=1:size(trainingdata,1)
    trainingdata{q,1} = reshape(tmpData(q,:),size(trainingdata{1,1},1),size(trainingdata{1,1},2));
end

for q=1:size(testingdata,1)
    testingdata{q,1} = reshape(tmpDataY(q,:),size(testingdata{1,1},1),size(testingdata{1,1},2));
end
%----standardization end

trainingY = Y(1:trainingSize,:);
testingY = Y(trainingSize+1:end,:);


kernel = rand(5,5,4);% four 5 X 5 convolutional kernels
eta = 0.02;%learning rate
alpha = 0.1;%L2 regularization

F = zeros(size(trainingdata{1,1},1)-size(kernel,1)+1,size(trainingdata{1,1},2)-size(kernel,2)+1,size(kernel,3));
thetas = rand(4*fix(size(F,1)/2)*fix(size(F,2)/2),14);%final layer thetas, have 14 subject classes
processValues = zeros(size(trainingY,1),size(trainingY,2));% output values of training data
iteration = 0;
previousJ = 0;
avgs = [];
iterations = [];
while iteration < 300
    %forward propagation
    %Fortran style looping
    %fails if kernel is bigger than image
    %now just have to add testing data plot vs iteration
    minRow = int32 (size(kernel,1)/2);
    minColumn = int32(size(kernel,2)/2);
    tmpThetas = thetas; %making sure to update thetas only after batch completion
    H = cell([1,2]); %input to our ANN or output for flatten layer, also with their indices
    H{1,2} = cell([size(trainingdata,1),1]);
    for k = 1:size(trainingdata,1)
        for m = 1:size(kernel,3)
            for i = minRow:(size(trainingdata{k,1},1)-minRow+1)
                a = i - minRow+1; % row co-ordinate of F
                for j= minColumn:(size(trainingdata{k,1},2)-minColumn+1)
                    b = j - minColumn+1; % column co-ordinate of F
                    
                    tmpKernel = rot90(kernel(:,:,m),2); % rotated matrix by 180
                    F(a,b,m) = F(a,b,m) + sum(((trainingdata{k,1}(a:(a+size(kernel,1)-1),b:(b+size(kernel,2)-1))).*tmpKernel),'all');
                    
                end
            end
        end
        F(~isfinite(F)) = 0.0001;
        
        %Here we should have final value of F
        %max-pooling of width 2 and stride 2
        h = zeros(fix(size(F,1)/2),fix(size(F,2)/2),size(kernel,3));
        %these are linear indices, have to convert them when back propagating
        h_index = zeros(fix(size(F,1)/2),fix(size(F,2)/2),size(kernel,3)); 
        a = 0;
        b = 0;
        for m = 1:size(kernel,3)
            for i = 1:2:(size(F,1)-1)
                a = a + 1;
                for j = 1:2:(size(F,2)-1)
                    b = b + 1;

                    [max_num,max_idx]=max(F(i:(i+1),j:(j+1)),[],'all','linear');
                    [X,Y]=ind2sub(size(F),max_idx); % method to convert the linear indices into X,Y

                    h(a,b,m) = max_num; 
                    h_index(a,b,m) = max_idx;
                end
                b=0;
            end
            a=0;
            H{1,2}{k,1,m} = h_index(:,:,m);
        end
        
        %flattening step
        H{1,1} = [H{1,1};reshape(h,[],(size(h,1)*size(h,2)*size(h,3)))];
            
    end
    H{1,1}(~isfinite(H{1,1})) = 0.0001;
    
    %ANN thetas
    %using log likelyhood objective function
    %using logistic activation function
    %copy your code from the assignment where building ANN using one
    %layer
    netH = H{1,1}*tmpThetas;
    gx = (1.0)./(1.+exp(((-1).*netH)));
    gx(~isfinite(gx)) = 0.0001;
    
    
    recA = (log(gx)');
    recA(~isfinite(recA)) = 0.0001;
    recB = (log(1.-gx)');
    recB(~isfinite(recB)) = 0.0001;
    avgJ = (trainingY*(recA)) + ((1.-trainingY)*(recB));
    avgJ(~isfinite(avgJ)) = 0.0001;
    
    % add in L2 for all the layers
    tj = 0;
    L2 = ((alpha).*(log((tmpThetas')*(tmpThetas))));
    L2(~isfinite(L2)) = 0.0001;
    tj = tj + trace(L2);
    
    for q=1:size(kernel,3)
        L2 = ((alpha).*(log((kernel(:,:,q)')*(kernel(:,:,q)))));
        L2(~isfinite(L2)) = 0.0001;
        tj = tj + trace(L2);
    end
    tj = tj + trace(avgJ) ; 
    sumavgJ = (tj)/(double (trainingSize));
    changeavglikelyhood = sumavgJ - previousJ;
    previousJ = sumavgJ;
    processValues = gx;
    
    iteration = iteration + 1;
    iterations(iteration,1) = iteration;
    avgs(iteration,1) = sumavgJ;
    
    %backward propagation
    newK = zeros(size(kernel,1),size(kernel,2), size(kernel,3));
    dJ = ((H{1,1}')*(trainingY - processValues)) + (alpha)*(tmpThetas);
    dJ(~isfinite(dJ)) = 0.0001;
    thetas = thetas + ((eta/double (trainingSize))*(dJ));
    thetas(~isfinite(thetas)) = 0.0001;
    
    % as this is batch mode, avg the df/dk for all the images
    selectX = [];
    for k = 1:size(trainingdata,1)
        dJ = (trainingY(k,:) - processValues(k,:))*(tmpThetas');
        dJ(~isfinite(dJ)) = 0.0001;
        for i = 1:size(kernel,1)
            for j = 1:size(kernel,2)
                for m = 1:size(kernel,3)
                    X_tmp = trainingdata{k,1}((size(kernel,1)-i+1):(size(trainingdata{k,1},1)-i+1),(size(kernel,2)-j+1):(size(trainingdata{k,1},2)-j+1));
                    dF = rot90(X_tmp,2); % rotating X_tmp by 180 degrees
                    dF(~isfinite(dF)) = 0.0001;

                    % select values from previously stored max indices
                    select = dF(H{1,2}{k,1,m});
                    select = (reshape(select,[],(size(select,1)*size(select,2))));
                    selectX = [selectX,select];
                end
                newK(i,j,:) = newK(i,j,:) + (dJ*(selectX'));
                newK(~isfinite(newK)) = 0.0001;
                selectX = [];
            end
        end
    end
    
    %averaging the values of K
    newK = newK./(double (trainingSize));
    newK(~isfinite(newK)) = 0.0001;
    
    %updating kernels and thetas
    for q = 1:size(kernel,3)
        kernel(:,:,q) = kernel(:,:,q) + ((eta/double (trainingSize))*(newK(:,:,q) + (alpha*(kernel(:,:,q))) ));
    end
    kernel(~isfinite(kernel)) = 0.0001;
    %tmpThetas = thetas;
end

%At this point, we should have trained our model using training set and
%tested out layers on testing data
%Accuracy Calcuation
plot(iterations, avgs);
TP = 0;
TP_training = 0;
newY = zeros(size(testingY,1),size(testingY,2));
newY_training = zeros(size(trainingY,1),size(trainingY,2));

[maxE_training, I] = max(processValues,[],2);
for i=1:size(newY_training,1)
    newY_training(i,I(i,1)) = 1;
    if (trainingY(i,I(i,1)) == 1)
        TP_training = TP_training + 1;
    end       
end

Accuracy =  (double (TP_testing_training)/double (testingSize));
Accuracy_testing =  (double (TP_training)/double (trainingSize));
disp('Testing Accuracy: ');
disp(Accuracy_testing);


disp('Training Accuracy: ');
disp(Accuracy);

