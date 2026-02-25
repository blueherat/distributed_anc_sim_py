classdef CircularArray < handle
    properties
        Radius      (1,1) double          % 半径 (米)
        NumElements (1,1) uint32          % 阵元数量
        Center      (1,3) double = [0,0,0]% 阵列中心坐标 [x, y, z]
        StartID     (1,1) uint32          % 起始 ID (例如 301)
    end

    properties (SetAccess = private)
        % 下列属性由 computeGeometry 自动计算，外部只读
        ElementAngles    (:,1) double % 每个阵元的角度 [0, 2pi)，逆时针
        ElementPositions (:,3) double % 每个阵元的笛卡尔坐标 [x, y, z]
        ElementIDs       (:,1) uint32 % ID 列表
    end

    methods
        function obj = CircularArray(radius, numElements, center, startID)
            % 构造函数
            arguments
                radius      (1,1) double
                numElements (1,1) uint32
                center      (1,3) double
                startID     (1,1) uint32
            end

            obj.Radius = radius;
            obj.NumElements = numElements;
            obj.Center = center;
            obj.StartID = startID;

            obj.computeGeometry();
        end

        function computeGeometry(obj)
            % 计算阵元的位置和角度
            % 策略：从 0 度开始 (x轴正方向)，逆时针均匀分布

            % 1. 生成角度向量 (0 ~ 2pi)
            % formula: phi_l = 2 * pi * (l - 1) / L
            obj.ElementAngles = 2 * pi / double(obj.NumElements) * double(0 : obj.NumElements - 1)';

            % 2. 生成 ID 列表
            obj.ElementIDs = obj.StartID + (0 : obj.NumElements - 1)';

            % 3. 计算笛卡尔坐标
            % x = cx + R * cos(phi)
            % y = cy + R * sin(phi)
            % z = cz
            x = obj.Center(1) + obj.Radius * cos(obj.ElementAngles);
            y = obj.Center(2) + obj.Radius * sin(obj.ElementAngles);
            z = obj.Center(3) * ones(obj.NumElements, 1);

            obj.ElementPositions = [x, y, z];
        end

        function registerToManager(obj, mgr, type)
            % 公共接口：根据类型分发调用
            arguments
                obj
                mgr
                type (1,1) string {mustBeMember(type, ["mic", "secondary"])}
            end

            switch type
                case "mic"
                    obj.registerMics(mgr);
                case "secondary"
                    obj.registerSpeakers(mgr);
            end
        end

    end

    methods (Access = private)
        function registerMics(obj, mgr)
            % 专门负责注册麦克风
            ids = obj.ElementIDs;
            pos = obj.ElementPositions;
            N = obj.NumElements;

            for i = 1:N
                mgr.addErrorMicrophone(ids(i), pos(i, :));
            end
        end

        function registerSpeakers(obj, mgr)
            % 专门负责注册次级扬声器
            ids = obj.ElementIDs;
            pos = obj.ElementPositions;
            N = obj.NumElements;

            for i = 1:N
                mgr.addSecondarySpeaker(ids(i), pos(i, :));
            end
        end
    end
end