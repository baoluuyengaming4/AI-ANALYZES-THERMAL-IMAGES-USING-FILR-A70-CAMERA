%% BƯỚC 1 : CHUẨN BỊ DỮ LIỆU ẢNH
disp('CHUẨN BỊ DỮ LIỆU ẢNH');
% THIẾT LẬP ĐƯỜNG DẪN THƯ MỤC ẢNH 
imageFolder = 'D:\FLIR\extracted_frames'; % Ảnh dùng để huấn luyện
outputCSV = 'thermal_dataset.csv';         % Dữ liệu trích xuất thư mục ảnh được lưu ở đây
disp('Dữ liệu đang được chuẩn bị....');
% LẤY DANH SÁCH FILE & SẮP XẾP THEO SỐ ĐƯỢC ĐÁNH TRONG TÊN 
fileList = dir(fullfile(imageFolder, '*.tif'));
fileNames = {fileList.name};

frameNumbers = zeros(length(fileNames), 1);
for k = 1:length(fileNames)
    % Trích số trong ngoặc: Train (12).tif → 12
    tokens = regexp(fileNames{k}, '\((\d+)\)\.tif$', 'tokens');
    if ~isempty(tokens)
        frameNumbers(k) = str2double(tokens{1}{1});
    else
        warning('⚠️ Không nhận dạng được số từ tên file: %s', fileNames{k});
        frameNumbers(k) = NaN;
    end
end

% Sắp xếp danh sách theo số frame
[~, sortIdx] = sort(frameNumbers);
fileList = fileList(sortIdx);
nFiles = length(fileList);

% KHỞI TẠO BẢNG KẾT QUẢ 
resultTable = table('Size', [nFiles, 4], ...
    'VariableTypes', {'datetime', 'double', 'double', 'double'}, ...
    'VariableNames', {'Timestamp', 'MaxTemp', 'MeanTemp', 'VarianceTemp'});

% DUYỆT FILE THEO THỨ TỰ 
for i = 1:nFiles
    fileName = fullfile(imageFolder, fileList(i).name);
    
    try
        thermalImage = double(imread(fileName));
    catch
        warning('⚠️ Không đọc được ảnh: %s', fileName);
        continue;
    end

    % Tính các thông số nhiệt độ
    maxTemp = max(thermalImage(:));
    meanTemp = mean(thermalImage(:));
    varTemp = var(thermalImage(:));
    
    % Tạm dùng timestamp giả lập: mỗi ảnh cách nhau 10 giây
    timestamp = datetime(i * 10, 'ConvertFrom', 'posixtime');

    % Ghi vào bảng
    resultTable.Timestamp(i) = timestamp;
    resultTable.MaxTemp(i) = maxTemp;
    resultTable.MeanTemp(i) = meanTemp;
    resultTable.VarianceTemp(i) = varTemp;
end

%  LƯU KẾT QUẢ 
writetable(resultTable, outputCSV);
disp('✅ Dữ liệu đã được trích xuất ');

% disp(outputCSV);

% %% BƯỚC 2 : CHUẨN BỊ DỮ LIỆU HUẤN LUYỆN CHO LSTM
% disp('Chuẩn bị dữ liệu huấn luyện cho LSTM ');
% % ĐỌC DỮ LIỆU :
% data = readtable('thermal_dataset.csv');
% maxTemps = data.MaxTemp;
% 
% %  Chuẩn các biến dữ liệu
% minVal = min(maxTemps);
% maxVal = max(maxTemps);
% normTemps = (maxTemps - minVal) / (maxVal - minVal); % [0,1]
% 
% % CẮT DỮ LIỆU THÀNH CHUỖI (multi-step)
% inputLen = 500;   % là số bước đầu vào
% outputLen = 270;  % là số bước cần dự đoán ( mỗi bước cách nhau 10s)
% 
% X = {}; Y = {};
% for i = 1:(length(normTemps) - inputLen - outputLen)
%     inputSeq = normTemps(i : i + inputLen - 1);
%     outputSeq = normTemps(i + inputLen : i + inputLen + outputLen - 1);
%     
%     X{end+1} = inputSeq';
%     Y{end+1} = outputSeq';
% end
% 
% % CHIA TẬP TRAIN/VALIDATION 
% N = numel(X);
% trainRatio = 0.8;
% idx = 1:floor(trainRatio * N);
% 
% XTrain = X(idx);
% YTrain = Y(idx);
% XVal   = X(idx(end)+1:end);
% YVal   = Y(idx(end)+1:end);
% 
% %LƯU DỮ LIỆU HUẤN LUYỆN 
% save('lstm_multistep_data.mat', 'XTrain', 'YTrain', 'XVal', 'YVal', 'minVal', 'maxVal');
% 
% disp('✅ Đã chuẩn bị dữ liệu huấn luyện cho LSTM (multi-step)');
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