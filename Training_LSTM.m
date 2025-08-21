%% BƯỚC 3 : BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH BĂNG MATHLAB
disp('BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH BĂNG MATHLAB ');
% Tải dữ liệu
load('lstm_multistep_data.mat');

% Lọc bỏ những mẫu có NaN trong X hoặc Y
validIdx = cellfun(@(x, y) all(~isnan(x)) && all(~isnan(y)), XTrain, YTrain);
XTrain = XTrain(validIdx);
YTrain = YTrain(validIdx);

% Xây dựng mô hình LSTM 
inputSize = 1;
numHiddenUnits = 100;
outputSize = outputLen;  % Số bước dự đoán

layers = [
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits, "OutputMode", "last")
    fullyConnectedLayer(outputSize)
    regressionLayer
];

options = trainingOptions("adam", ...
    "MaxEpochs", 150, ...
    "MiniBatchSize", 64, ...
    "Shuffle", "every-epoch", ...
    "Plots", "training-progress", ...
    "Verbose", false);

%  Huấn luyện 
% Chuyển YTrain từ cell array sang ma trận (mỗi hàng là 1 chuỗi dự đoán)
YTrainMatrix = cell2mat(YTrain(:));

% Kiểm tra khớp số mẫu
assert(numel(XTrain) == size(YTrainMatrix, 1), ...
    "❌ Số mẫu trong XTrain và YTrain không khớp!");

net = trainNetwork(XTrain, YTrainMatrix, layers, options);

% Đánh giá trên tập validation ===
% Chuẩn hóa ngược lại (denormalize)
YPred = predict(net, XVal, "MiniBatchSize", 1);
yTrue = YVal{1} * (maxVal - minVal) + minVal;
yPred = YPred(1, :) * (maxVal - minVal) + minVal;
% Vẽ biểu đồ đề mô đánh giá chất lượng mô hình train 
figure;
plot(yTrue, '-ob'); hold on;
plot(yPred, '--or');
legend('Thực tế (°C)', 'Dự đoán (°C)');
title('Dự đoán nhiệt độ LSTM');
xlabel('Bước thời gian');
ylabel('Nhiệt độ (°C)');
grid on;

%  Lưu mô hình
save('lstm_model.mat', 'net', 'minVal', 'maxVal');
disp('✅ Đã huấn luyện và lưu mô hình LSTM.');