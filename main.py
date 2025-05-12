import torch
def drop_column(A: torch.tensor) -> torch.tensor:
    """
    Drops the column of A containing the most 0 values.

    Parameters:
        tensor: A 2D tensor.
    Returns:
        tensor: The input tensor with the column containing the most 0 values removed.
    """

    result = None
    ### your code here ###
    # Find the column with the most zeros
    n_rows = A.size()[0]
    n_cols = A.size()[1]
    
    # cols_with_zeros = torch.sum(A == 0, dim=0)
    zero_counts = (A == 0).sum(dim=0)
    col_to_remove = torch.argmax(zero_counts)
    mask = torch.arange(A.size(1)) != col_to_remove
    print(mask)
    # col_with_most_zeros = torch.where(A == 0)
    # print(col_with_most_zeros)
    
    print(result)
    # Drop that column
    # result = torch.cat([A[:, :col_with_most_zeros[1][0]], A[:, col_with_most_zeros[1][0] + 1:]], dim=1)
    # return result
if __name__ == "__main__":
    x = torch.tensor([[0, 0, 7],
        [0, 0, 1],
        [3, 0, 5]])
    # print(x)
    print(drop_column(x))
    