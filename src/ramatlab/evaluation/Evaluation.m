classdef Evaluation
    properties
        % Instance of the Python Evaluation class
        evaluationInstance
    end
    
    methods
        function obj = Evaluation()
            % Constructor: Initializes an instance of the Evaluation class in Python
            % Usage:
            %   wrapper = Evaluation();
            obj.evaluationInstance = py.importlib.import_module('src.rapython.evaluation.evaluation').Evaluation();
        end

        function map = evalMeanAveragePrecision(obj, test_data_loc, rel_data_loc, topk, loader)
            map = obj.evaluationInstance.eval_mean_average_precision(test_data_loc, rel_data_loc, py.int(topk), loader);
        end

        function rank = evalRank(obj, test_data_loc, rel_data_loc, topk, loader)
            rank = obj.evaluationInstance.eval_rank(test_data_loc, rel_data_loc, py.int(topk), loader);
        end

        function recall = evalRecall(obj, test_data_loc, rel_data_loc, topk, loader)
            recall = obj.evaluationInstance.eval_recall(test_data_loc, rel_data_loc, py.int(topk), loader);
        end

        function precision = evalPrecision(obj, test_data_loc, rel_data_loc, topk, loader)
            precision = obj.evaluationInstance.eval_precision(test_data_loc, rel_data_loc, py.int(topk), loader);
        end

        function ndcg = evalNDCG(obj, test_data_loc, rel_data_loc, topk, loader)
            ndcg = obj.evaluationInstance.eval_ndcg(test_data_loc, rel_data_loc, py.int(topk), loader);
        end

    end
end
