function [params, spec, sens, acc, auc,relevant_ind,aucXtst,aucYtst] = eye_train2(wX0, wX1,aX0, aX1,perc_threshold, num_of_pca)
%
% X0: [Nsamples0 * Nfeats]
% X1: [Nsamples1 * Nfeats]
%

N0 = max(size(wX0, 1),size(aX0, 1));
N1 = max(size(wX1, 1),size(aX1, 1));

wX = zscore([wX0; wX1]');
wX = wX';
aX = zscore([aX0; aX1]');
aX = aX';
% wX = [wX0; wX1];
% aX = [aX0; aX1];
Y = [ones(1,N0) 2*ones(1,N1)]';

nfolds = 5;
CV = cvpartition(Y, 'k', nfolds);
meanAucX = linspace(0,1,ceil(size(Y,1)/5));
meanAucY = zeros(size(meanAucX));

relevant_ind = [];
for i = 1:CV.NumTestSets
    trIdx = CV.training(i);
    tstIdx = CV.test(i);
    Ytr = Y(trIdx, :);
    Ytst = Y(tstIdx, :);
    aXtst = [];aXtr = []; wXtst = []; wXtr = [];
    if ~isempty(wX)
        wXtr = wX(trIdx, :);
        wXtst = wX(tstIdx, :);
        relevant_ind(i,:) = find_meaningful_feats(wXtr,Ytr,perc_threshold);
        wXtr = wXtr(:,relevant_ind(i,:));
%         [wXtr,wmu,wsigma]=zscore(wXtr);
        
        [wpca,wXtr] = princomp(wXtr);
        wXtr = wXtr(:,1:num_of_pca);
     
        wXtst = wXtst(:,relevant_ind(i,:));
%         wXtst = (wXtst - repmat(wmu,size(wXtst,1),1)) ./ repmat(wsigma,size(wXtst,1),1);
        wXtst = wXtst*wpca(:,1:num_of_pca);
        
    end
    if ~isempty(aX)
        aXtr=aX(trIdx, :);
%         [aXtr,amu,asigma]=zscore(aXtr);
        [apca,aXtr] = princomp(aXtr);
        aXtr = aXtr(:,1:num_of_pca);
        aXtst = aX(tstIdx, :);
%         aXtst = (aXtst - repmat(amu,size(aXtst,1),1)) ./ repmat(asigma,size(aXtst,1),1);
        aXtst = aXtst*apca(:,1:num_of_pca); 
%         aXtr = aX(trIdx, :);
%         aXtst = aX(tstIdx, :);
    end
    
%     Xtr = cat(2,wXtr,aXtr);
%     [Xtr, mu, sigma] = zscore(Xtr);
%     Xtst = cat(2,wXtst,aXtst);
%     Xtst = (Xtst - repmat(mu,size(Xtst,1),1)) ./ repmat(sigma,size(Xtst,1),1);

    wXtr =zscore(wXtr');
    wXtr = wXtr';
    aXtr =zscore(aXtr');
    aXtr = aXtr';
    Xtr = cat(2,wXtr,aXtr);
    
    wXtst =zscore(wXtst');
    wXtst = wXtst';
    aXtst =zscore(aXtst');
    aXtst = aXtst';
    Xtst = cat(2,wXtst,aXtst);
    


    clearvars wXtr wXtst aXtr aXtst
    
        
    N0tr = sum(Ytr == 1);
    N1tr = sum(Ytr == 2);
    N0tst = sum(Ytst == 1);
    N1tst = sum(Ytst == 2);

    % train
    obj = train_shrinkage(Xtr, Ytr);
    W(:,:,i) = obj.W;
    %W(:,:,i) = rand(size(obj.W))*0.02-0.01;
    
    % calc acc on train sample
    Q = Xtr*W(:,:,i);    
    Q0 = Q(Ytr == 1);
    Q1 = Q(Ytr == 2);
    
    ths = Q + eps;
    ths = sort(ths);
    for k = 1:length(ths)                
        sens(k) = length(find(Q1 <= ths(k))) / N1tr;
        spec(k) = length(find(Q0 > ths(k))) / N0tr;    
    end;
    idx = find(spec >= 0.99, 1, 'last');
    th_opt(i) = ths(idx); 
    sens_tr(i) = sens(idx);
    spec_tr(i) = spec(idx);    
    [~, ~, ~, auc_tr(i)] = perfcurve([ones(N1tr,1); zeros(N0tr,1)], [Q1; Q0], 0);
    
    % test
    Q = Xtst*W(:,:,i);
    clearvars Xtst Xtr
    Q0 = Q(Ytst == 1);
    Q1 = Q(Ytst == 2);
    sens_tst(i) = length(find(Q1 <= th_opt(i))) / N1tst;
    spec_tst(i) = length(find(Q0 > th_opt(i))) / N0tst;    
     
    [aucX,aucY, ~, auc_tst(i)] = perfcurve([ones(N1tst,1); zeros(N0tst,1)], [Q1; Q0], 0);
    disp(sprintf('Sens %f\n\t',sens_tst(i)));
    disp(sprintf('Spec %f\n\t', spec_tst(i)));
    disp(sprintf('Auc %f\n\t',auc_tst(i)));
    [aucX, index]=unique(aucX);
    meanAucY = meanAucY + interp1(aucX,aucY(index),meanAucX);
end

spec.tr = [mean(spec_tr) std(spec_tr)];
sens.tr = [mean(sens_tr) std(sens_tr)];
acc.tr = 0;
spec.tst = [mean(spec_tst) std(spec_tst)];
sens.tst = [mean(sens_tst) std(sens_tst)];
acc.tst = 0;
auc.tr = 0;
auc.tst = [mean(auc_tst) std(auc_tst)];
auc.all = 0;
aucXtst = meanAucX;
aucYtst = meanAucY/nfolds;


params.W = mean(W, 3);
params.th = mean(th_opt);

% fid = fopen('../res/acc3.txt', 'w');
% fprintf(fid, 'Train:\n');
% fprintf(fid, ' Sensitivity: %f +- %f\n', sens.tr);
% fprintf(fid, ' Specificity: %f +- %f\n', spec.tr);
% fprintf(fid, '\n');
% fprintf(fid, 'Test:\n');
% fprintf(fid, ' Sensitivity: %f +- %f\n', sens.tst);
% fprintf(fid, ' Specificity: %f +- %f\n', spec.tst);
% fclose(fid);

