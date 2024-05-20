
class NoLicensePlateException(Exception):
    def __init__(self):
        self.message = 'No license plate detected in the image.'
        super().__init__(self.message)


class HoughLinesException(Exception):
    def __init__(self, ver_lines: int, hor_lines: int):
        self.message = f'Not enough lines detected. Vertical lines: {ver_lines}, Horizontal lines: {hor_lines}.'
        super().__init__(self.message)


class NoCharactersException(Exception):
    def __init__(self):
        self.message = 'No characters detected in the image.'
        super().__init__(self.message)