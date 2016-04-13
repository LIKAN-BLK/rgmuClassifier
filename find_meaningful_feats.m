function [ind] = find_meaningful_feats(X,y,perc_threshold)
%find_meaningdul_feats Find indexes of features
%  Finds features, that correlate with labels using r^2 metrics 
% We chosing feature when it r2 metric lager then r2 metrics of
% perc_threshold another features. Return mask wich delete unrelevant
% features
%-------1-dimensional------------------------------

r2 = zeros(1,size(X,2));
for i = 1:size(X,2)
    tmp = corrcoef(X(:,i),y);
    r2(i) = (tmp(1,2))^2;
end
threshold = prctile(r2,perc_threshold);
ind = find(r2>threshold);

%-----3-dimensional---------------------------------
% r2 = zeros(size(X,1),size(X,2),size(X,3));
% for sc = 1:size(X,1)
%     for t = 1:size(X,2)
%         for ch = 1:size(X,3)
%             tmp = corrcoef(X(sc,t,ch,:),y);
%             r2(sc,t,ch) = (tmp(1,2))^2;
%         end
%     end
% end
% threshold = prctile(r2(:),perc_threshold);
% ind = find(r2>threshold);
% [scale,time,channel] = ind2sub(size(r2),find(r2>threshold));






