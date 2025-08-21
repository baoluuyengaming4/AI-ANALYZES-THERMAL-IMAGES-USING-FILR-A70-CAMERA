%% BƯỚC 2 : CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN CHO LSTM
disp('Chuẩn bị dữ liệu huấn luyện cho LSTM ');
% ĐỌC DỮ LIỆU :
data = readtable('thermal_dataset.csv');
maxTemps = data.MaxTemp;

%  Chuẩn các biến dữ liệu
minVal = min(maxTemps);
maxVal = max(maxTemps);
normTemps = (maxTemps - minVal) / (maxVal - minVal); % [0,1]

% CẮT DỮ LIỆU THÀNH CHUỖI (multi-step)
inputLen = 500;   % là số bước đầu vào
outputLen = 270;  % là số bước cần dự đoán ( mỗi bước cách nhau 10s 270 bước tương ứng với 45p)

X = {}; Y = {};
for i = 1:(length(normTemps) - inputLen - outputLen)
    inputSeq = normTemps(i : i + inputLen - 1);
    outputSeq = normTemps(i + inputLen : i + inputLen + outputLen - 1);
    
    X{end+1} = inputSeq';
    Y{end+1} = outputSeq';
end

% CHIA TẬP TRAIN/VALIDATION 
N = numel(X);
trainRatio = 0.8;
idx = 1:floor(trainRatio * N);

XTrain = X(idx);
YTrain = Y(idx);
XVal   = X(idx(end)+1:end);
YVal   = Y(idx(end)+1:end);

%LƯU DỮ LIỆU HUẤN LUYỆN 
save('lstm_multistep_data.mat', 'XTrain', 'YTrain', 'XVal', 'YVal', 'minVal', 'maxVal');

disp('✅ Đã chuẩn bị dữ liệu huấn luyện cho LSTM (multi-step)');