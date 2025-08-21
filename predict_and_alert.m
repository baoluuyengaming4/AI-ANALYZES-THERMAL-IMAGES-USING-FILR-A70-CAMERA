% === DỰ BÁO & CẢNH BÁO CÓ ƯỚC TÍNH THỜI GIAN (RUL) ===
% cứ 10p thì sẽ dự đoán xu hướng nhiệt độ trong 45p tới ( Thiết lập tần suất giám sát cao cho
% theo dõi dõi biến đông nhiệt độ nhanh có thể thay đổi tần suất dự đoán nếu nhiệt độ thay đổi chậm)
%  cập nhật nhiệt độ mới mỗi 5 phút 1 lần sẽ theo dõi biến động ngắn hạn (Tăng hay giảm ) tiện cho 
% việc kiểm tra và theo dõi trực tiếp
% ( tính chênh lệch nhiệt độ giữa 2 lần đo)


% Tải mô hình và dữ liệu chuẩn hóa
load lstm_model.mat net minVal maxVal
load lstm_multistep_data.mat 
inputLen = 500;   % Số bước đầu vào
outputLen = 270;  % Số bước đầu ra (mỗi bước cách nhau 10s => 270 bước là 45 phút)

% Tham số theo dõi ngắn
trackingInterval = 300; % 5 phút = 300 giây
checkCount = 0;
trackedTemps = [];

while true
    % Đọc toàn bộ ảnh từ  thư mục
    imageFolder = 'D:/FLIR/New images';
    fileList = dir(fullfile(imageFolder, '*.tif'));
    
    % Sắp xếp ảnh theo thời gian
    [~, idx] = sort([fileList.datenum]);
    fileList = fileList(idx);

    if length(fileList) < inputLen
        error('Không đủ ảnh để dự đoán. Cần ít nhất %d ảnh.', inputLen);
    end

    % Lấy các ảnh gần nhất
    recentFiles = fileList(end - inputLen + 1:end);
    inputTemps = zeros(inputLen, 1);
    for i = 1:inputLen
        img = double(imread(fullfile(imageFolder, recentFiles(i).name)));
        inputTemps(i) = max(img(:));
    end

    % Ghi lại nhiệt độ mới nhất
    currentTemp = inputTemps(end);
    trackedTemps(end+1) = currentTemp;
    checkCount = checkCount + 1;

    fprintf('\n== THỜI GIAN: %s ==\n', datestr(now));
    fprintf('🌡️ Nhiệt độ hiện tại: %.2f°C\n', currentTemp);

    % Kiểm tra xu hướng nhiệt độ thay đổi nếu có đủ 2 lần đo
    if checkCount >= 2
        deltaTemp = trackedTemps(end) - trackedTemps(end-1);
        if deltaTemp > 0.5
            fprintf('📈 Xu hướng: Nhiệt độ tăng (Δ = %.2f°C trong 10 phút qua).\n', deltaTemp);
        elseif deltaTemp < -0.5
            fprintf('📉 Xu hướng: Nhiệt độ giảm (Δ = %.2f°C trong 10 phút qua).\n', deltaTemp);
        else
            fprintf('➡️ Xu hướng: Nhiệt độ ổn định (Δ = %.2f°C trong 10 phút qua).\n', deltaTemp);
        end
    end

    % Dự đoán sau mỗi 2 lần theo dõi (tức 10 phút)
    if checkCount >= 2
        % Chuẩn hóa input
        inputNorm = (inputTemps - minVal) / (maxVal - minVal);
        XTest = {inputNorm'};

        % Dự đoán 270 bước
        YPred = predict(net, XTest, 'MiniBatchSize', 1);
        yPred = YPred(1, :) * (maxVal - minVal) + minVal;

        % Phân tích xu hướng và cảnh báo
        maxPred = max(yPred);
        rul_step = find(yPred > 95, 1);
        rul_step_m1 = find(yPred > 82, 1);

        if maxPred > 95
            if isempty(rul_step)
                fprintf('🔥 [MỨC 2] Nhiệt độ sẽ > 95°C nhưng chưa xác định rõ thời gian. Kiểm tra khẩn cấp!\n');
            else
                rul_sec = rul_step * 10;
                fprintf('🔥 [MỨC 2] Dự đoán nhiệt độ %.2f°C sẽ vượt 95°C sau %d giây (~%.1f phút). Kiểm tra khẩn cấp!\n', maxPred, rul_sec, rul_sec/60);
            end
        elseif maxPred > 82
            if isempty(rul_step_m1)
                fprintf('⚠️ [MỨC 1] Nhiệt độ %.2f°C > 82°C nhưng chưa xác định thời gian. Theo dõi sát.\n', maxPred);
            else
                rul_sec = rul_step_m1 * 10;
                fprintf('⚠️ [MỨC 1] Dự đoán nhiệt độ %.2f°C > 82°C sau %d giây (~%.1f phút). Theo dõi sát.\n', maxPred, rul_sec, rul_sec/60);
            end
        else
            fprintf('✅ Bình thường. Nhiệt độ đỉnh dự đoán: %.2f°C trong 45 phút tới.\n', maxPred);
        end

                % Ghi log đầy đủ
        logFile = 'log_du_doan.csv';
        current_time = datetime('now');
        newRow = table(current_time, currentTemp, deltaTemp, maxPred, ...
            'VariableNames', {'Timestamp', 'CurrentTemp', 'DeltaTemp_5min', 'PredictedMaxTemp_45min'});
        
        if isfile(logFile)
            writetable(newRow, logFile, 'WriteMode', 'append');
        else
            writetable(newRow, logFile);
        end
        % Reset bộ đếm
        checkCount = 0;
        trackedTemps = [];
    end

    pause(trackingInterval);  % Chờ 5 phút
end