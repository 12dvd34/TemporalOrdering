from functools import total_ordering


# TODO: add support for era
@total_ordering
class Date:
    def __init__(self, year, month=0, day=0):
        self.year = year
        self.month = month
        self.day = day

    def __str__(self):
        return str(self.year) + "-" + str(self.month) + "-" + str(self.day)

    def __lt__(self, other):
        if self.year < other.year:
            return True
        elif self.year > other.year:
            return False
        elif self.month < other.month:
            return True
        elif self.month > other.month:
            return False
        elif self.day < other.day:
            return True
        else:
            return False

    def __eq__(self, other):
        if self.year != other.year:
            return False
        elif self.month != other.month:
            return False
        elif self.day != other.day:
            return False
        else:
            return True
