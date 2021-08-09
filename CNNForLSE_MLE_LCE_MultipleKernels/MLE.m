clear all; %remove all the old variables in the workspace
close all;

Y0 = zeros(40,40);
Y1 = zeros(40,40);
Y = [1;0];
Y0(:,20) = 255*ones(40,1); %Vertical line
Y1(20,:) = 255*ones(1,40); %Horizontal line
data = cell([2,1]);
data{1,1} = Y0;
data{2,1} = Y1;
rng(2,'philox');
kernel = (0.000001)*rand(40,40); % could be multiplied by 0.01 to get interesting results
initial = kernel;
F = zeros(size(data{1,1},1)-size(kernel,1)+1,size(data{1,1},2)-size(kernel,2)+1);
thetas = (0.000001)*rand(fix(size(F,1)/1)*fix(size(F,2)/1),1);
trainingSize = 2;
iterations = [];
avgs = [];
iteration = 0;
eta = 0.013;%learning rate
previousJ = 0;
alpha = 0.001;%L2 regularization
while iteration <=200
    %forward propagation
    %Fortran style looping
    %fails if kernel is bigger than image
    %now just have to add testing data plot vs iteration
    minRow = int32 (size(kernel,1)/2);
    minColumn = int32(size(kernel,2)/2);
    tmpThetas = thetas; %making sure to update thetas only after batch completion
    H = cell([1,2]); %input to our ANN or output for flatten layer, also with their indices
    H{1,2} = cell([size(data,1),1]);
    for k = 1:size(data,1)
        for m = 1:size(kernel,3)
            for i = minRow:(size(data{k,1},1)-minRow) %look for odd and even for +/-1
                a = i - minRow+1; % row co-ordinate of F
                for j= minColumn:(size(data{k,1},2)-minColumn)
                    b = j - minColumn+1; % column co-ordinate of F
                    
                    tmpKernel = rot90(kernel(:,:,m),2); % rotated matrix by 180
                    F(a,b) = F(a,b) + sum(((data{k,1}(a:(a+size(kernel,1)-1),b:(b+size(kernel,2)-1))).*tmpKernel),'all');
                    
                end
            end
        end
        F(~isfinite(F)) = 0.0001;
        
        %Here we should have final value of F
        %max-pooling of width 1 and stride 1
        h = zeros(fix(size(F,1)/1),fix(size(F,2)/1));
        %these are linear indices, have to convert them when back propagating
        h_index = zeros(fix(size(F,1)/1),fix(size(F,2)/1)); 
        a = 0;
        b = 0;
        for i = 1:1:(size(F,1))
            a = a + 1;
            for j = 1:1:(size(F,2))
                b = b + 1;
                
                [max_num,max_idx]=max(F,[],'all','linear');
                [X,Y_max]=ind2sub(size(F),max_idx); % method to convert the linear indices into X,Y
                
                h(a,b) = max_num; 
                h_index(a,b) = max_idx;
            end
            b=0;
        end
        
        %flattening step
        %should you add bias feature or not
        H{1,1} = [H{1,1};reshape(h,[],(size(h,1)*size(h,2)))];
        H{1,2}{k,1} = h_index;
    end
    H{1,1}(~isfinite(H{1,1})) = 0.0001;
    
    %ANN thetas
    %using log likelihood objective function
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
    avgJ = (Y*(recA)) + ((1.-Y)*(recB));
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
    newK = zeros(size(kernel,1),size(kernel,2));
    dJ = (((H{1,1}'))*(Y-processValues)) + (alpha)*(tmpThetas);
    dJ(~isfinite(dJ)) = 0.0001;
    thetas = thetas + ((eta/double (trainingSize))*(dJ));
    thetas(~isfinite(thetas)) = 0.0001;
    
    % as this is batch mode, avg the df/dk for all the images 
    for k = 1:size(data,1)
        dJ = (Y(k,:) - processValues(k,:))*(tmpThetas');
        dJ(~isfinite(dJ)) = 0.0001;
        for i = 1:size(kernel,1)
            for j = 1:size(kernel,2)
                %rowsizeMin = (size(kernel,1)-i+1);
                %rowsizeMax = (size(data{k,1},1)-i+1);
                
                %columnsizeMin = (size(kernel,2)-j+1);
                %columnsizeMax = (size(data{k,2},1)-j+1);
                X_tmp = data{k,1}((size(kernel,1)-i+1):(size(data{k,1},1)-i+1),(size(kernel,2)-j+1):(size(data{k,1},2)-j+1));
                dF = rot90(X_tmp,2); % rotating X_tmp by 180 degrees
                dF(~isfinite(dF)) = 0.0001;
                
                % select values from previously stored max indices
                select = dF(H{1,2}{k,1});
                newK(i,j) = newK(i,j) + (dJ*((reshape(select,[],(size(select,1)*size(select,2))))'));
                newK(~isfinite(newK)) = 0.0001;
            end
        end
    end
    
    %averaging the values of K
    newK = newK./(double (trainingSize));
    newK(~isfinite(newK)) = 0.0001;
    
    %updating kernels
    for q = 1:size(kernel,3)
        kernel(:,:,q) = kernel(:,:,q) + ((eta/double (trainingSize))*(newK + (alpha*(kernel(:,:,q))) ));
    end
    kernel(~isfinite(kernel)) = 0.0001;
    %tmpThetas = thetas;

end

final = kernel;






subplot(2,2,1),imshow(initial);
title('Initial Kernel')


subplot(2,2,2),imshow(final);
title('Final Kernel');



subplot(2,2,[3,4]);
plot(iterations, avgs);
title('log likelihood vs Iteration Number');
xlabel('Iteration Number');
ylabel('log likelihood');
%savefig('ANN2.fig');
