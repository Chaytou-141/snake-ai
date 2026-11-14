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
