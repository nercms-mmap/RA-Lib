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

        function recall = computeRecall(obj, data_list, rel_list, topk, input_type)
            % Calculates recall score based on data and relevance lists
            % Parameters:
            %   data_list (array): List of data scores or rankings
            %   rel_list (array): List of relevant items
            %   topk (int): The number of top-ranked items to consider
            %   input_type (string): Specifies the format of data_list (e.g., 'score' or 'rank')
            % Returns:
            %   recall (double): Calculated recall score
            recall = obj.evaluationInstance.compute_recall(data_list, rel_list, py.int(topk), input_type);
        end
        
        function precision = computePrecision(obj, score_list, rel_list, topk, input_type)
            % Calculates precision score based on score and relevance lists
            % Parameters:
            %   score_list (array): List of data scores
            %   rel_list (array): List of relevant items
            %   topk (int): The number of top-ranked items to consider
            %   input_type (string): Specifies the format of score_list (e.g., 'score')
            % Returns:
            %   precision (double): Calculated precision score
            precision = obj.evaluationInstance.compute_precision(score_list, rel_list, py.int(topk), input_type);
        end

        function rank = computeRank(obj, list_data, rel_list, topk, input_type)
            % Calculates ranking score based on data and relevance lists
            % Parameters:
            %   list_data (array): List of data rankings or scores
            %   rel_list (array): List of relevant items
            %   topk (int): The number of top-ranked items to consider
            %   input_type (string): Specifies the format of list_data (e.g., 'rank')
            % Returns:
            %   rank (double): Calculated ranking score
            rank = obj.evaluationInstance.compute_rank(list_data, rel_list, py.int(topk), input_type);
        end

        function ap = computeAveragePrecision(obj, score_list, rel_list, topk, input_type)
            % Computes Average Precision (AP) score for a set of predictions
            % Parameters:
            %   score_list (array): List of scores
            %   rel_list (array): List of relevance labels
            %   topk (int): The number of top items to consider
            %   input_type (string): Format of input list (e.g., 'score')
            % Returns:
            %   ap (double): Computed average precision score
            ap = obj.evaluationInstance.compute_average_precision(score_list, rel_list, py.int(topk), input_type);
        end

        function map = evalMeanAveragePrecisionDF(obj, test_data, rel_data, topk)
            % Evaluates Mean Average Precision (MAP) for DataFrame inputs
            % Parameters:
            %   test_data (DataFrame): Testing data with scores
            %   rel_data (DataFrame): Ground truth relevance data
            %   topk (int): Number of top items to consider in the evaluation
            % Returns:
            %   map (double): Computed mean average precision
            map = obj.evaluationInstance.eval_mean_average_precision(test_data, rel_data, py.int(topk));
        end

        function map = evalMeanAveragePrecisionMAT(obj, test_path, rel_path, test_data_name, test_rel_name, data_type, topk)
            % Evaluates Mean Average Precision (MAP) for .mat file inputs
            % Parameters:
            %   test_path (string): Path to test data .mat file
            %   rel_path (string): Path to relevance data .mat file
            %   test_data_name (string): Variable name for test data in .mat file
            %   test_rel_name (string): Variable name for relevance data in .mat file
            %   data_type (string): Type of data (e.g., 'score')
            %   topk (int): Number of top items to consider
            % Returns:
            %   map (double): Computed mean average precision
            map = obj.evaluationInstance.eval_mean_average_precision(test_path, rel_path, test_data_name, test_rel_name, data_type, py.int(topk));
        end

        function rank = evalRankDF(obj, test_data, rel_data, topk)
            % Evaluates rank score for DataFrame inputs
            % Parameters:
            %   test_data (DataFrame): Testing data with scores
            %   rel_data (DataFrame): Ground truth relevance data
            %   topk (int): Number of top items to consider in evaluation
            % Returns:
            %   rank (double): Computed rank score
            rank = obj.evaluationInstance.eval_rank(test_data, rel_data, py.int(topk));
        end

        function rank = evalRankMAT(obj, test_path, rel_path, test_data_name, test_rel_name, data_type, topk)
            % Evaluates rank score for .mat file inputs
            % Parameters:
            %   test_path (string): Path to test data .mat file
            %   rel_path (string): Path to relevance data .mat file
            %   test_data_name (string): Variable name for test data in .mat file
            %   test_rel_name (string): Variable name for relevance data in .mat file
            %   data_type (string): Type of data (e.g., 'rank')
            %   topk (int): Number of top items to consider in evaluation
            % Returns:
            %   rank (double): Computed rank score
            rank = obj.evaluationInstance.eval_rank(test_path, rel_path, test_data_name, test_rel_name, data_type, py.int(topk));
        end

        function recall = evalRecallDF(obj, test_data, rel_data, topk)
            % Evaluates recall score for DataFrame inputs
            % Parameters:
            %   test_data (DataFrame): Testing data with scores
            %   rel_data (DataFrame): Ground truth relevance data
            %   topk (int): Number of top items to consider
            % Returns:
            %   recall (double): Computed recall score
            recall = obj.evaluationInstance.eval_recall(test_data, rel_data, py.int(topk));
        end

        function recall = evalRecallMAT(obj, test_path, rel_path, test_data_name, test_rel_name, data_type, topk)
            % Evaluates recall score for .mat file inputs
            % Parameters:
            %   test_path (string): Path to test data .mat file
            %   rel_path (string): Path to relevance data .mat file
            %   test_data_name (string): Variable name for test data in .mat file
            %   test_rel_name (string): Variable name for relevance data in .mat file
            %   data_type (string): Data type, e.g., 'score'
            %   topk (int): Number of top items to consider
            % Returns:
            %   recall (double): Computed recall score
            recall = obj.evaluationInstance.eval_recall(test_path, rel_path, test_data_name, test_rel_name, data_type, py.int(topk));
        end

        function precision = evalPrecisionDF(obj, test_data, rel_data, topk)
            % Evaluates precision score for DataFrame inputs
            % Parameters:
            %   test_data (DataFrame): Testing data with scores
            %   rel_data (DataFrame): Ground truth relevance data
            %   topk (int): Number of top items to consider
            % Returns:
            %   precision (double): Computed precision score
            precision = obj.evaluationInstance.eval_precision(test_data, rel_data, py.int(topk));
        end

        function dcg = computeDCG(obj, rank_list, rel_list, topk)
            % Computes Discounted Cumulative Gain (DCG) score
            % Parameters:
            %   rank_list (array): List of rankings
            %   rel_list (array): List of relevance labels
            %   topk (int): Number of top items to consider
            % Returns:
            %   dcg (double): Computed DCG score
            dcg = obj.evaluationInstance.compute_dcg(rank_list, rel_list, py.int(topk));
        end

        function ndcg = computeNDCG(obj, list_data, rel_list, topk, input_type)
            % Computes Normalized Discounted Cumulative Gain (NDCG) score
            % Parameters:
            %   list_data (array): List of scores or ranks
            %   rel_list (array): List of relevance labels
            %   topk (int): Number of top items to consider
            %   input_type (string): Data format (e.g., 'rank')
            % Returns:
            %   ndcg (double): Computed NDCG score
            ndcg = obj.evaluationInstance.compute_ndcg(list_data, rel_list, py.int(topk), input_type);
        end

        function ndcg = evalNDCGDF(obj, test_data, rel_data, topk)
            % Evaluates Normalized Discounted Cumulative Gain (NDCG) for DataFrame inputs
            % Parameters:
            %   test_data (DataFrame): Testing data with scores
            %   rel_data (DataFrame): Ground truth relevance data
            %   topk (int): Number of top items to consider
            % Returns:
            %   ndcg (double): Computed NDCG score
            ndcg = obj.evaluationInstance.eval_ndcg(test_data, rel_data, py.int(topk));
        end

        function ndcg = evalNDCGMAT(obj, test_path, rel_path, test_data_name, test_rel_name, data_type, topk)
            % Evaluates Normalized Discounted Cumulative Gain (NDCG) for .mat file inputs
            % Parameters:
            %   test_path (string): Path to test data .mat file
            %   rel_path (string): Path to relevance data .mat file
            %   test_data_name (string): Variable name for test data in .mat file
            %   test_rel_name (string): Variable name for relevance data in .mat file
            %   data_type (string): Data type, e.g., 'score'
            %   topk (int): Number of top items to consider
            % Returns:
            %   ndcg (double): Computed NDCG score
            ndcg = obj.evaluationInstance.eval_ndcg(test_path, rel_path, test_data_name, test_rel_name, data_type, py.int(topk));
        end
    end
end
