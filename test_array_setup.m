%% test_array_setup.m
% 脚本用于测试 CircularArray 的生成、以及通过 RIRManager 进行统一可视化和 RIR 计算
clear; clc; close all;

% 导入包
import acoustics.*

%% 1. 基础参数设置
fprintf('=== 1. 初始化参数 ===\n');
Fs = 8000;
room_dims = [5, 5, 4]; % 房间大小 [x, y, z]
center = room_dims / 2; % 阵列放置在房间中心

% 几何参数
R_mic = 0.3;     % 麦克风半径
L = 7;           % 麦克风数量
mic_start_id = 301;

R_spk = 0.6;     % 次级扬声器半径
Q = 7;           % 扬声器数量
spk_start_id = 201;

%% 2. 创建阵列对象 & RIRManager
fprintf('=== 2. 生成对象并注册设备 ===\n');
mgr = RIRManager();
mgr.Room = room_dims;
mgr.Fs = Fs;
mgr.MaterialAbsorption = 0.5;

% 2.1 创建阵列几何
micArr = CircularArray(R_mic, uint32(L), center, uint32(mic_start_id));
spkArr = CircularArray(R_spk, uint32(Q), center, uint32(spk_start_id));

fprintf('麦克风阵列: %d 个阵元, 半径 %.2f m\n', micArr.NumElements, micArr.Radius);
fprintf('扬声器阵列: %d 个阵元, 半径 %.2f m\n', spkArr.NumElements, spkArr.Radius);

% 2.2 注册主声源 (手动)
primary_pos = center + [1.5, 0.5, 0];
mgr.addPrimarySpeaker(101, primary_pos);

% 2.3 注册阵列 (使用新接口)
micArr.registerToManager(mgr, "mic");
spkArr.registerToManager(mgr, "secondary");

fprintf('所有设备已注册到 RIRManager。\n');

%% 3. 统一可视化布局 (使用新功能)
fprintf('=== 3. 绘制 3D 布局图 ===\n');
figure('Name', 'System Layout Check', 'Color', 'w', 'Position', [100, 100, 800, 600]);
ax = axes();

% --- 调用 RIRManager 的 plotLayout 方法 ---
% 这会自动画出房间轮廓、主声源(红)、次级声源(绿)、麦克风(蓝)
mgr.plotLayout(ax); % true 表示显示 ID 标签

% 调整视角为俯视，方便检查圆阵
view(ax, 2);
title(ax, 'Simulation Layout (Top View)');
drawnow;

fprintf('请检查图形窗口：\n  - 红色方块: 主声源\n  - 绿色菱形: 次级扬声器 (外圈)\n  - 蓝色圆圈: 误差麦克风 (内圈)\n');

%% 4. 计算 RIR
fprintf('=== 4. 计算 RIR (耗时操作) ===\n');
tic;
mgr.build(true); % verbose=true 显示进度
elapsed = toc;
fprintf('RIR 计算完成，耗时: %.4f 秒\n', elapsed);

%% 5. 验证生成的数据
fprintf('=== 5. 数据验证与绘图 ===\n');

% 随机选取一条路径：第1个次级扬声器 -> 第1个麦克风
test_spk_id = spkArr.ElementIDs(1);
test_mic_id = micArr.ElementIDs(1);

try
    h = mgr.getSecondaryRIR(test_spk_id, test_mic_id);

    figure('Name', 'RIR Verification', 'Color', 'w', 'Position', [150, 150, 800, 400]);

    % 时域图
    subplot(1, 2, 1);
    t_axis = (0:length(h)-1) / Fs;
    plot(t_axis, h);
    title(sprintf('Impulse Response\n(Spk %d -> Mic %d)', test_spk_id, test_mic_id));
    xlabel('Time (s)'); ylabel('Amplitude'); grid on;

    % 频域图
    subplot(1, 2, 2);
    H = fft(h, 1024);
    freq_axis = (0:511) * Fs / 1024;
    plot(freq_axis, 20*log10(abs(H(1:512)) + eps));
    title('Frequency Response');
    xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)'); grid on;

    fprintf('验证成功：成功提取并绘制了 RIR。\n');

catch ME
    fprintf('验证失败：提取 RIR 时出错。\n错误信息: %s\n', ME.message);
end