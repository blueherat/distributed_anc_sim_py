classdef RIRManager < handle
    properties
        Room (1,3) double = [7 5 3]
        Fs (1,1) double = 48000
        SoundSpeed (1,1) double = 343

        Algorithm (1,1) string = "image-source"

        ImageSourceOrder (1,1) double = 3

        NumStochasticRays (1,1) double = 1280
        MaxNumRayReflections (1,1) double = 10
        ReceiverRadius (1,1) double = 0.5

        BandCenterFrequencies = [125 250 500 1000 2000 4000]
        AirAbsorption = 0
        MaterialAbsorption = ...
            [0.10 0.20 0.40 0.60 0.50 0.60; ... %地面
            0.10 0.20 0.40 0.60 0.50 0.60; ... %前墙
            0.10 0.20 0.40 0.60 0.50 0.60; ... %后墙
            0.10 0.20 0.40 0.60 0.50 0.60; ... %左墙
            0.02 0.03 0.03 0.03 0.04 0.07; ... %右墙
            0.02 0.03 0.03 0.03 0.04 0.07].';  %天花
        MaterialScattering = []

        PrimarySpeakers   dictionary
        SecondarySpeakers dictionary
        ErrorMicrophones  dictionary
    end

    properties (Access = private)
        PrimaryRIRs   dictionary
        SecondaryRIRs dictionary
    end

    methods
        function obj = RIRManager()
            % 初始化 dictionary
            obj.PrimarySpeakers   = configureDictionary("uint32", "cell");
            obj.SecondarySpeakers = configureDictionary("uint32", "cell");
            obj.ErrorMicrophones  = configureDictionary("uint32", "cell");

            obj.PrimaryRIRs       = configureDictionary("string", "cell");
            obj.SecondaryRIRs     = configureDictionary("string", "cell");
        end

        function addPrimarySpeaker(obj, id, position)
            arguments
                obj
                id (1,1) uint32
                position (1,3) double
            end
            obj.PrimarySpeakers(id) = {position};
        end

        function addSecondarySpeaker(obj, id, position)
            arguments
                obj
                id (1,1) uint32
                position (1,3) double
            end
            obj.SecondarySpeakers(id) = {position};
        end

        function addErrorMicrophone(obj, id, position)
            arguments
                obj
                id (1,1) uint32
                position (1,3) double
            end
            obj.ErrorMicrophones(id) = {position};
        end

        function build(obj, verbose)
            if nargin < 2, verbose = true; end

            if numEntries(obj.ErrorMicrophones) == 0
                error('No error microphones have been added.');
            end

            micIds = keys(obj.ErrorMicrophones);
            micPositions = cell2mat(values(obj.ErrorMicrophones));
            % --- 1. 主通路 ---
            if numEntries(obj.PrimarySpeakers) > 0
                for spkId = keys(obj.PrimarySpeakers)'
                    spkPos = obj.PrimarySpeakers(spkId);
                    ir = obj.computeRIR(spkPos, micPositions);

                    for j = 1:numel(micIds)
                        micId = micIds(j);
                        key = "P" + string(spkId) + "->M" + string(micId);
                        obj.PrimaryRIRs(key) = {ir(j, :)};
                    end
                    if verbose
                        disp("Primary paths for Speaker " + spkId + " computed. RIR length: " + size(ir,2));
                    end
                end
            else
                error('No primary speakers have been added.');
            end
            % --- 2. 次级通路 ---
            if numEntries(obj.SecondarySpeakers) > 0
                for spkId = keys(obj.SecondarySpeakers)'
                    spkPos = obj.SecondarySpeakers(spkId);
                    ir = obj.computeRIR(spkPos, micPositions);

                    for j = 1:numel(micIds)
                        micId = micIds(j);
                        key = "S" + string(spkId) + "->M" + string(micId);
                        obj.SecondaryRIRs(key) = {ir(j, :)};
                    end
                    if verbose
                        disp("Secondary paths for Speaker " + spkId + " computed. RIR length: " + size(ir,2));
                    end
                end
            else
                error('No secondary speakers have been added.');
            end
        end

        function h = getPrimaryRIR(obj, spkId, micId)
            key = "P" + string(spkId) + "->M" + string(micId);
            if ~isKey(obj.PrimaryRIRs, key)
                error('Primary RIR for path (%s) does not exist.', key);
            end
            h = cell2mat(obj.PrimaryRIRs(key));
        end

        function h = getSecondaryRIR(obj, spkId, micId)
            key = "S" + string(spkId) + "->M" + string(micId);
            if ~isKey(obj.SecondaryRIRs, key)
                error('Secondary RIR for path (%s) does not exist.', key);
            end
            h = cell2mat(obj.SecondaryRIRs(key));
        end

        function d = calculateDesiredSignal(obj, referenceSignal, nSamples)
            keyPriSpks = keys(obj.PrimarySpeakers);
            keyErrMics = keys(obj.ErrorMicrophones);
            numPriSpks = numEntries(obj.PrimarySpeakers);
            numErrMics = numEntries(obj.ErrorMicrophones);
            x = referenceSignal;

            d = zeros(nSamples, numErrMics); % 期望信号
            for m = 1:numErrMics
                for j = 1:numPriSpks
                    P = obj.getPrimaryRIR(keyPriSpks(j), keyErrMics(m));
                    d_jm = conv(x(:, j), P);
                    d(:, m) = d(:, m) + d_jm(1:nSamples);
                end
            end
        end

        function plotLayout(obj, ax)
            % PLOTLAYOUT 可视化房间及所有声学设备布局
            % ax: (可选) 绘图坐标区句柄

            arguments
                obj
                ax = []
            end

            if isempty(ax)
                figure('Name', 'Acoustic Layout', 'Color', 'w');
                ax = gca;
            end

            hold(ax, 'on');
            grid(ax, 'on');
            axis(ax, 'equal');
            xlabel(ax, 'X (m)'); ylabel(ax, 'Y (m)'); zlabel(ax, 'Z (m)');
            title(ax, 'Room Layout Configuration');

            % --- 1. 绘制房间轮廓 ---
            L = obj.Room(1); W = obj.Room(2); H = obj.Room(3);

            % 直接定义包含 NaN 断点的坐标序列 (底面 -> 顶面 -> 4条棱)
            x_nodes = [0 L L 0 0   NaN   0 L L 0 0   NaN   0 0 NaN L L NaN L L NaN 0 0];
            y_nodes = [0 0 W W 0   NaN   0 0 W W 0   NaN   0 0 NaN 0 0 NaN W W NaN W W];
            z_nodes = [0 0 0 0 0   NaN   H H H H H   NaN   0 H NaN 0 H NaN 0 H NaN 0 H];

            plot3(ax, x_nodes, y_nodes, z_nodes, 'k--', 'LineWidth', 1, 'DisplayName', 'Room Boundary');

            % --- 2. 绘制主声源 (红色正方形) ---
            obj.plotDevices(ax, obj.PrimarySpeakers, 'r', 's', 'Primary Source');

            % --- 3. 绘制次级声源 (绿色菱形) ---
            obj.plotDevices(ax, obj.SecondarySpeakers, 'g', 'd', 'Secondary Source');

            % --- 4. 绘制误差麦克风 (蓝色圆圈) ---
            obj.plotDevices(ax, obj.ErrorMicrophones, 'b', 'o', 'Error Mic');

            % --- 5. 设置视图 ---
            view(ax, 3); % 默认 3D 视图
            legend(ax, 'Location', 'bestoutside');

            % 限制坐标轴范围略大于房间
            xlim(ax, [-0.5, L+0.5]);
            ylim(ax, [-0.5, W+0.5]);
            zlim(ax, [0, H+0.5]);
        end
    end

    methods (Access = private)
        function ir = computeRIR(obj, txPos, rxPos)
            ir = acousticRoomResponse( ...
                obj.Room, txPos{1}, rxPos, ...
                SampleRate=obj.Fs, SoundSpeed=obj.SoundSpeed, ...
                Algorithm=obj.Algorithm, ...
                ImageSourceOrder=obj.ImageSourceOrder, ...
                NumStochasticRays=obj.NumStochasticRays, ...
                MaxNumRayReflections=obj.MaxNumRayReflections, ...
                ReceiverRadius=obj.ReceiverRadius, ...
                AirAbsorption=obj.AirAbsorption, ...
                MaterialAbsorption=obj.MaterialAbsorption, ...
                MaterialScattering=obj.MaterialScattering, ...
                BandCenterFrequencies=obj.BandCenterFrequencies);
        end

        function plotDevices(~, ax, deviceDict, color, marker, labelName)
            % 辅助函数：绘制特定类型的设备并配置数据游标 (Data Tip)
            
            if numEntries(deviceDict) > 0
                ids = keys(deviceDict);
                % 提取位置矩阵 (N x 3)
                positions = cell2mat(values(deviceDict)); 
                
                % 1. 绘制散点
                s = scatter3(ax, positions(:,1), positions(:,2), positions(:,3), ...
                    60, color, 'filled', ...
                    'Marker', marker, ...
                    'DisplayName', labelName);
                
                % 2. 配置 Data Tip
                s.DataTipTemplate.DataTipRows = [
                    s.DataTipTemplate.DataTipRows(1:3); % X, Y, Z
                    dataTipTextRow('ID', double(ids));  % ID
                ];
            end
        end
    end
end