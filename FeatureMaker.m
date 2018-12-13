
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Author: Maral Kasiri, Sepehr Jalali,
% This code provides the feature set of the wavelet transform coeffiecients.
% Used for the Neural Network Input and also for the statistical feature
% extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reading the main dataset and making a set of images out of it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[label1 , mi1]= MakeImage('female_1.mat');    
[label2 , mi2]= MakeImage('female_2.mat');
[label3 , mi3]= MakeImage('female_3.mat');
[label4 , mi4]= MakeImage('male_1.mat');
[label5 , mi5]= MakeImage('male_2.mat');
[label6 , mi6]= MakeImage('male_day_1.mat');
[label7 , mi7]= MakeImage('male_day_2.mat');
[label8 , mi8]= MakeImage('male_day_3.mat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Concatenating the whole dataset 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data= cat(1, mi1(:,1:2500,:), mi6, mi3(:,1:2500,:),mi7, mi5(:,1:2500,:),mi4(:,1:2500,:), mi2(:,1:2500,:), mi8);
Label= cat(1, label1,label6, label3, label7,label5, label4, label2,label8);
Data1= permute(Data,[1,3,2]);
save('RawDataCnn.mat', 'Data1', 'Label')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% making the features from the raw data: the features
% arethe wavelet transform coefficients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

featureCNN=MakeFeatureCNN(Data,Label);
save('CNN_Feature2','featureCNN')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Feature_gb=ExtractFeature(Data,Label);
save('Feature_GB.mat', 'Feature_gb')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function featureVector= ExtractFeature(Data, Label)  %%% Feature maker for GB

[m,n,k]= size(Data);
feature_set= zeros(m,47);
for j=1:2
    for i=1:m
    [c,l]= wavedec(Data(i,:,j),5,'db4');
    approx=appcoef(c,l,'db4');
    [cd1,cd2,cd3,cd4, cd5]= detcoef(c,l,[1 2 3 4 5]);
    MAbsVal= [mean(abs(cd1)), mean(abs(cd2)),mean(abs(cd2)),mean(abs(cd2)),mean(abs(cd2)),mean(abs(approx))];
    AvPower= [bandpower(cd1),bandpower(cd2),bandpower(cd3),bandpower(cd4),bandpower(cd5),bandpower(approx)];
    Sdev= [std(cd1),std(cd2), std(cd3),std(cd4),std(cd5),std(approx)];
    Ratio= [abs(mean(cd2))/ abs(mean(cd1)), abs(mean(cd3))/ abs(mean(cd2)), abs(mean(cd4))/ abs(mean(cd3)), abs(mean(cd5))/ abs(mean(cd4)), abs(mean(approx))/ abs(mean(cd5))];
    
    if j==1
        feature_set(i,1:24)= [Label(i,1) MAbsVal AvPower Sdev Ratio];
    else
        feature_set(i,25:end)= [MAbsVal AvPower Sdev Ratio];
    end
    
    end
end
   
featureVector=feature_set;
end

function FeatureSetCNN= MakeFeatureCNN(data,label)    %%%%% Feature vector generator

[m,n,k]= size(data);
for j=1:2
for i=1:m
    [c,l]= wavedec(data(i,:,j),4,'db2');
    approx=appcoef(c,l,'db2');
    [cd1,cd2,cd3,cd4]= detcoef(c,l,[1 2 3 4]);
    n=length(c);
    if j==1
        feature_set(i,1:n+1)=[label(i,1) cd1 cd2 cd3 cd4 approx];
    else 
        feature_set(i, n+2:2*n+1)=[cd1 cd2 cd3 cd4 approx];
    end
end
end
FeatureSetCNN=feature_set;
end



function [label, makeImage]= MakeImage(filename)  %%%%% make images from the original dataset + labels

load(filename);
cyl= cat(3,cyl_ch1, cyl_ch2);
hook= cat(3, hook_ch1, hook_ch2);
lat= cat(3, lat_ch1, lat_ch2);
palm= cat(3,palm_ch1, palm_ch2);
spher= cat(3, spher_ch1, spher_ch2);
tip= cat(3, tip_ch1, tip_ch2);



makeImage= cat(1, cyl, hook, lat, palm, spher, tip);

[m,n,k]= size(cyl);
L=ones(m,1);
label= [0.*L ; 1.*L ; 2.*L; 3.*L; 4.*L ; 5.*L];
end