"""Static board geometry, independent of rendering and machine learning."""


def free_cell_distance(game, direction) -> int:
    """Free cells before the first wall/body cell; current tail is static here."""
    occupied = set(game.snake)
    dx, dy = direction.value
    x, y = game.snake[0]
    distance = 0
    while True:
        x, y = x + dx, y + dy
        if not (0 <= x < game.width and 0 <= y < game.height) or (x, y) in occupied:
            return distance
        distance += 1


def body_free_cell_distance(game, direction) -> int:
    """Static body-only ray: free cells before body, or fixed maximum if clear.

    The current tail is occupied. Walls end visibility but are not body hits;
    a ray with no visible body returns max(width, height)-1 (normalised 1).
    """
    distance = free_cell_distance(game, direction)
    dx, dy = direction.value
    head_x, head_y = game.snake[0]
    first_blocked = (head_x + (distance + 1) * dx, head_y + (distance + 1) * dy)
    return distance if first_blocked in game.snake[1:] else max(game.width, game.height) - 1
