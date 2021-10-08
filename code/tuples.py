import re
from typing import NamedTuple

import numpy as np


class Country(NamedTuple):
    name:str
    def __str__(self):
        return self.name

class CPC(NamedTuple):
    section: str
    class_: str
    subclass: str
    group: str
    subgroup: str

    def __str__(self):
        return f'{self.section}{self.class_}{self.subclass} {self.group}/{self.subgroup}'

    def at_level(self, k):
        return CPC(*[cpc if i < k else '*' for i, cpc in enumerate(self)])

    @staticmethod
    def parse(l):
        if not isinstance(l, str):
            return None
        p = re.compile(r'.*([A-H|Y])(\d{2})([A-Z]) (\d+)/(\d+).*')
        return CPC(*p.match(l).groups())


class Patent(NamedTuple):
    t: np.datetime64
    id: str
    i:int

