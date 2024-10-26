%{
    IRA - Class implementing the Iterative Rank Aggregation (IRA) algorithm for ranking tasks.

    This class provides a MATLAB interface to the Python IRA implementation, enabling the execution
    of the IRA method for ranking aggregation based on user-defined parameters.

    Reference:
    ----------
    - Huang, J., Liang, C., Zhang, Y., Wang, Z., & Zhang, C. (2022). Ranking Aggregation with Interactive Feedback for Collaborative Person Re-identification. In BMVC (p. 386).

    Author:
    -------
    Qi Deng

    Date:
    -----
    2024-10-18
%}

classdef IRA
    methods(Static)
        function ira(input_file_path, output_file_path, input_rel_path, k_set, iteration, error_rate, mode, input_type)
            % ira - Executes the Iterative Rank Aggregation (IRA) using specified input and output file paths.
            %
            % This function loads data from CSV files, processes the data into a numerical format, and
            % applies the appropriate IRA method based on the specified mode.
            %
            % Parameters:
            % -----------
            % input_file_path : str
            %     Path to the input CSV file containing query, voter name, item code, and item rank/score.
            %
            % output_file_path : str
            %     Path to the output CSV file where the ranked results will be saved.
            %
            % input_rel_path : str
            %     Path to the input CSV file containing query, item code, and relevance.
            %
            % k_set : int
            %     The number of top items to consider for feedback and updating the ranks.
            %
            % iteration : int
            %     The number of iterations to run the re-ranking process.
            %
            % error_rate : float, optional
            %     The interaction error rate that simulates uncertainty in feedback. Default is 0.02.
            %
            % mode : MethodType, optional
            %     The mode of operation to determine which IRA method to use (RANK or SCORE). Default is MethodType.IRA_RANK.
            %
            % input_type : InputType, optional
            %     The type of input data to determine how the item rank/score is interpreted (RANK or SCORE). Default is InputType.SCORE.
            %
            % Returns:
            % --------
            % None
            %     The function saves the ranked results directly to the specified output file path.

            % Call the Python ira function
            py.importlib.import_module('src.rapython.supervised.ira').ira(input_file_path, output_file_path, input_rel_path, py.int(k_set), py.int(iteration), error_rate, mode, input_type);
        end
    end
end
