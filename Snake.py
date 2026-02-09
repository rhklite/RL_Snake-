def update_snake_length(block, snake_list):
    """
    Adds the new block to the front of the snake list, increasing its length by 1.
    Args:
        block: The new head position (tuple or list, e.g., (x, y)).
        snake_list: The current list representing the snake's body.
    Returns:
        The updated snake list with increased length.
    """
    return [block] + snake_list
