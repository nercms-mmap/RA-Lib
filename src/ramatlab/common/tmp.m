function tmp()


% 导入 Python 模块
constants = py.importlib.import_module('constant').InputType;


% 直接访问枚举成员
rank_value = py.importlib.import_module('src.rapython.common.constant').InputType.RANK;
score_value = py.importlib.import_module('src.rapython.common.constant').InputType.SCORE;

% 显示枚举值
disp(['RANK: ', char(rank_value)]);
disp(['SCORE: ', char(score_value)]);
end