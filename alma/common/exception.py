class AlmaException(Exception):
    class_message = "An exception occurred."

    def __init__(self, message=None, **kwargs):
        if message:
            self._message = message % kwargs
        else:
            self._message = self.class_message % kwargs

    def __repr__(self):
        return self._message

    def __str__(self):
        return self._message


class SizeMustBeEqual(AlmaException):
    class_message = ("Vector size must be equal. "
                     "Current sizes are not: "
                     "%(size1)s != %(size2)s")
