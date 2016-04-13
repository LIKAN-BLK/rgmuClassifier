
clearvars
addpath('..\..\work4\');
useAmpl = 1;
useWavelet = 1;
% result_path = ['.\resultsAmp' num2str(useAmpl) 'Wave' num2str(useWavelet) '\'];

hamster=[1 2 6 7 8 9 10 11];%[1 2 6 7 8 9 10 11];
sensivity = [];
specifity = [];
roc = [];
for pnum = [80]
    result_path = ['.\NormINtrialsNormOUTtrialsAmpl' num2str(useAmpl) 'Wave' num2str(useWavelet) 'PCA' num2str(pnum)  '\'];
    mkdir(result_path);
    fileID = fopen([result_path 'result.txt'],'wb');
    fprintf(fileID,['exp thres sens spec auc\r\n']);
    fclose(fileID);
    for h=hamster
        wXt = [];
        wXnt = [];

        aXt = [];
        aXnt = [];
        threshold = 1;
        expTitle = sprintf('4%02d',h);
        data_path = ['..\mat\wLets\' expTitle '\'];
        if useWavelet
            T = load([data_path 'wEEG_Te' expTitle '.mat'],'-mat');
            NT = load([data_path 'wEEG_NTe' expTitle '.mat'],'-mat'); 
            tmp_mask = load([data_path 'wEEG_info.mat'],'-mat');

            mask = NaN(size(tmp_mask.mask));

            %calcilate scale (X) to freq (y) transofrm X*coef = y
            low_fq = 5; high_fq = 30;
            X = [1,1;size(mask,1),1];
            y = [high_fq;low_fq];
            coeff = X\y;

            %decimate in dependence from frequency
            epoch_len = 0.5; % seconds 
            for sc = 1:size(mask,1)
                fq = [sc,1]*coeff; % corresponding frequency
                num_of_samples = fq*epoch_len*4; %minimal num of samples - fq * epoch_len. If we need more - multiply on something
                step = round(size(mask,2)/num_of_samples);
                mask(sc,1:step:end) = 1; 
            end

            mask = mask.*tmp_mask.mask;
            clear tmp_mask

            T = T.w(:,:,:,:);
            NT = NT.nw(:,:,:,:);
            for ch=1:size(T,3)
                for trial=1:size(T,4)
                    T(:,:,ch,trial) = T(:,:,ch,trial)+mask;
                end
                for trial=1:size(NT,4)
                    NT(:,:,ch,trial) = NT(:,:,ch,trial)+mask;
                end
            end
            clear mask
            Nt = size(T,4);
            Nnt = size(NT,4);


            % Unroll to 2d (trial x data) matrix
            wXt = zeros(Nt,size(T,1)*size(T,2)*size(T,3));
            for i = 1:Nt
                tmp = T(:,:,:,i);
                wXt(i,:) = tmp(:);
            end

            wXnt = zeros(Nnt,size(NT,1)*size(NT,2)*size(NT,3));
            for i = 1:Nnt
                tmp = NT(:,:,:,i);
                wXnt(i,:) = tmp(:);
            end
            clearvars T NT
            threshold = [70]; %[99.6,99.7,99.8,99.9,99.99];
            wXnt = wXnt(:,~isnan(wXnt(1,:)));
            wXt = wXt(:,~isnan(wXt(1,:)));
        end
        if useAmpl
            % Load amp features
            aXt = load([data_path 'aEEG_Te' expTitle '.mat'],'-mat');
            aXnt = load([data_path 'aEEG_NTe' expTitle '.mat'],'-mat'); 
            aXt=aXt.X1;
            aXnt=aXnt.X0;
        end
        %---------------------------------------
        indexes=cell(1,size(threshold,2));
        for i=1:size(threshold,2)
            [params, spec, sens, acc, auc,indexes{i},aucX,aucY] = eye_train2(wXnt,wXt,aXnt,aXt,threshold(i),pnum);
            fileID = fopen([result_path 'result.txt'],'a');
            fprintf(fileID,'%s %f %f %f %f\r\n',expTitle, threshold(i), sens.tst(1), spec.tst(1),auc.tst(1));
            fclose(fileID);
            plot(aucX,aucY);
            hold on;
            im=plot(0:0.1:1,0:0.1:1);
            hold off;
            saveas(im,[result_path expTitle 'roc.png'])
            saveas(im,[result_path expTitle 'roc.fig'])
            sensivity = [sensivity sens.tst(1)];
            specifity = [specifity spec.tst(1)];
            roc = [roc auc.tst(1)];
        end

    %     if useWavelet
    %         for i=1:size(indexes,2)
    %             tmp=indexes{i};
    %             res_mat = [];
    %             for j=1:size(tmp,1)
    %                 [sc,t,ch]=ind2sub(size(wXt(:,:,:,1)),tmp(j,:));
    %                 res_mat(:,:,j) = [sc;t;ch];
    %             end
    %             save([result_path expTitle '_' num2str(threshold(i)) '.mat'],'res_mat');        
    %         end
    %     end
    end
    fileID = fopen([result_path 'result.txt'],'a');
    fprintf(fileID,'Mean - %f %f %f\r\n',mean(sensivity), mean(specifity),mean(roc));
    fprintf(fileID,'STD - %f %f %f\r\n',std(sensivity), std(specifity),std(roc));
    
    fclose(fileID);
end
% fclose(fileID);
