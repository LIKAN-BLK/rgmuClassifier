function [params, spec, sens, acc, auc,relevant_ind] = eye_train1(wX0, wX1,aX0, aX1,perc_threshold, num_of_pca)
%
% X0: [Nsamples0 * Nfeats]
% X1: [Nsamples1 * Nfeats]
%

N0 = max(size(wX0, 1),size(aX0, 1));
N1 = max(size(wX1, 1),size(aX1, 1));

wX = [wX0; wX1];
aX = [aX0; aX1];
Y = [ones(1,N0) 2*ones(1,N1)]';


CV = cvpartition(Y, 'k', 5);


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
        [wpca,wXtr] = princomp(wXtr);
        wXtr = wXtr(:,1:num_of_pca);
     
        wXtst = wXtst(:,relevant_ind(i,:));
        wXtst = wXtst*wpca(:,1:num_of_pca);
        
    end
    if ~isempty(aX)
       [apca,aXtr] = princomp(aX(trIdx, :));
        aXtr = aXtr(:,1:num_of_pca);
        aXtst = aX(tstIdx, :)*apca(:,1:num_of_pca);               
    end
    Xtr = cat(2,wXtr,aXtr);
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
    
    % calc threshold using all sample
    if ~isempty(wX0) && ~isempty(aX0)
        Q = [wX(:,relevant_ind(i,:))*wpca(:,1:num_of_pca),aX*apca(:,1:num_of_pca)]*W(:,:,i);
    end
    if ~isempty(wX0) && isempty(aX0)
        Q = wX(:,relevant_ind(i,:))*wpca(:,1:num_of_pca)*W(:,:,i);
    end
    if  isempty(wX0) && ~isempty(aX0)
        Q = aX*apca(:,1:num_of_pca)*W(:,:,i);
    end
    Q0 = Q(Y == 1); %non target
    Q1 = Q(Y == 2); %target
    ths = Q + eps;
    ths = sort(ths);
    for k = 1:length(ths)                
        sens_all(k) = length(find(Q1 <= ths(k))) / N1;
        spec_all(k) = length(find(Q0 > ths(k))) / N0;    
    end;
    idx = find(spec_all >= 0.95, 1, 'last');
    %[~, idx] = max((sens_all * N1 + spec_all * N0) / (N1 + N0));
    th_opt(i) = ths(idx);
    [~, ~, ~, auc0(i)] = perfcurve([ones(N1,1); zeros(N0,1)], [Q1; Q0], 1);
    auc0(i) = max([auc0(i) 1-auc0(i)]);
    
    % calc acc on train sample
    Q = Xtr*W(:,:,i);    
    Q0 = Q(Ytr == 1);
    Q1 = Q(Ytr == 2);
    sens_tr(i) = length(find(Q1 <= th_opt(i))) / N1tr;
    spec_tr(i) = length(find(Q0 > th_opt(i))) / N0tr;    
    acc_tr(i) = (sens_tr(i) * N1tr + spec_tr(i) * N0tr) / (N1tr + N0tr);    
    [~, ~, ~, auc_tr(i)] = perfcurve([ones(N1tr,1); zeros(N0tr,1)], [Q1; Q0], 1);
    auc_tr(i) = max([auc_tr(i) 1-auc_tr(i)]);
    
%     ths = Q + eps;
%     ths = sort(ths);
%     for k = 1:length(ths)                
%         sens_tr(k) = length(find(Q1 <= ths(k))) / N1tr;
%         spec_tr(k) = length(find(Q0 > ths(k))) / N0tr;    
%     end;
%     %[acc_tr(freq, t, ch), idx] = max((sens_tr * N1tr + spec_tr * N0tr) / (N1tr + N0tr));
%     idx = find(spec_tr >= 0.97, 1, 'last');
%     spec_tr_opt(i) = spec_tr(idx);
%     sens_tr_opt(i) = sens_tr(idx);
%     acc_tr_opt(i) = (sens_tr_opt(i) * N1tst + spec_tr_opt(i) * N0tst) / (N1tst + N0tst);
%     th_opt = ths(idx);
    
    % test
    Q = Xtst*W(:,:,i);
    clearvars Xtst Xtr
    Q0 = Q(Ytst == 1);
    Q1 = Q(Ytst == 2);
    sens_tst(i) = length(find(Q1 <= th_opt(i))) / N1tst;
    spec_tst(i) = length(find(Q0 > th_opt(i))) / N0tst;    
    acc_tst(i) = (sens_tst(i) * N1tst + spec_tst(i) * N0tst) / (N1tst + N0tst);
    [~, ~, ~, auc_tst(i)] = perfcurve([ones(N1tst,1); zeros(N0tst,1)], [Q1; Q0], 1);
    auc_tst(i) = max([auc_tst(i) 1-auc_tst(i)]);
    %plot(1-spec_tr, sens_tr);    
end

spec.tr = [mean(spec_tr) std(spec_tr)];
sens.tr = [mean(sens_tr) std(sens_tr)];
acc.tr = [mean(acc_tr) std(acc_tr)];
spec.tst = [mean(spec_tst) std(spec_tst)];
sens.tst = [mean(sens_tst) std(sens_tst)];
acc.tst = [mean(acc_tst) std(acc_tst)];
auc.tr = [mean(auc_tr) std(auc_tr)];
auc.tst = [mean(auc_tst) std(auc_tst)];
auc.all = [mean(auc0) std(auc0)];

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